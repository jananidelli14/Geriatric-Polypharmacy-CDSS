"""
RAG System for Clinical Decision Support
Integrates a curated medical knowledge base with Gemini API.
"""

import os
import logging
from typing import List, Dict, Tuple
import re

import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MedicalRAG:
    """Retrieval-Augmented Generation for Medical Knowledge"""

    def __init__(self, persist_directory: str = "medical_knowledge_db", reset: bool = False):
        """
        Initialize RAG system with persistence.
        reset=False -> reuse existing collection (default)
        reset=True  -> delete and rebuild from scratch
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.persist_directory = os.path.join(current_dir, persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)

        logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        collection_name = "medical_knowledge"

        if reset:
            try:
                self.client.delete_collection(collection_name)
                logger.info("Existing collection deleted (reset=True).")
            except Exception:
                logger.info("No existing collection to delete (first run).")

        # Load embedding model
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully.")

        # Create or get collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(
                f"Loaded existing collection '{collection_name}' "
                f"with {self.collection.count()} documents."
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection '{collection_name}'.")

    # INGESTION
    def ingest_documents(self, documents: List[Dict[str, str]]):
        """Ingest medical documents into vector database."""
        if not documents:
            logger.warning("No documents provided for ingestion.")
            return

        logger.info(f"Ingesting {len(documents)} medical knowledge snippets...")

        ids = []
        embeddings = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            text = doc["text"]
            embedding = self.embedding_model.encode(text)

            ids.append(f"doc_{i}_{doc.get('category', 'general')}")
            embeddings.append(embedding.tolist())
            texts.append(text)
            metadatas.append({
                "source": doc.get("source", "Unknown"),
                "category": doc.get("category", "general"),
                "priority": doc.get("priority", 1),
                "topic": doc.get("topic", "General"),
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(
            f"Successfully ingested {len(documents)} documents into collection "
            f"'{self.collection.name}'."
        )
        logger.info(f"Database location: {self.persist_directory}")

    # RETRIEVAL
    def retrieve_context(self, query: str, top_k: int = 5) -> Tuple[List[str], List[Dict]]:
        """Retrieve relevant medical context for a query."""
        if self.collection.count() == 0:
            logger.warning("Knowledge base is empty!")
            return [], []

        query_embedding = self.embedding_model.encode(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        logger.info(f"Retrieved {len(documents)} relevant documents for RAG.")
        return documents, metadatas

    # GENERATION
    def generate_rag_response(
        self,
        user_query: str,
        gemini_model,
        conversation_context: str = "",
        top_k: int = 5,
    ) -> Dict:
        """Generate RAG-enhanced response using Gemini."""

        documents, metadatas = self.retrieve_context(user_query, top_k)

        if not documents:
            logger.warning("No relevant documents found for RAG query.")
            return {
                "response": (
                    "No specific clinical guidelines were found in the knowledge base for this query. "
                    "Please consult current clinical references and practice guidelines."
                ),
                "sources": [],
                "confidence": "low",
                "retrieved_docs": 0,
            }

        # Build context with clear source numbering
        context_text = "\n=== RETRIEVED MEDICAL KNOWLEDGE ===\n"
        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            source = meta.get("source", "Unknown")
            topic = meta.get("topic", "General")
            context_text += f"\n[Source {i}: {source} - {topic}]\n{doc}\n"

        system_prompt = """You are a medical reference assistant providing EDUCATIONAL information only.

CRITICAL RULES:
1. You provide INFORMATION, not ADVICE or INSTRUCTIONS
2. Use phrases like "research indicates", "guidelines suggest", "evidence shows"
3. NEVER use directive language: no "should", "must", "avoid", "recommend"
4. ALWAYS cite sources using [1], [2], [3] format
5. Keep response to 5-7 concise bullet points
6. NO preamble, NO disclaimers, NO introductory text - ONLY bullet points
7. Each bullet must be 1-2 sentences maximum
8. Focus on: drug safety, interactions, monitoring, renal considerations"""

        full_prompt = f"""{system_prompt}

{context_text}

USER QUERY: {user_query}

Provide ONLY 5-7 bullet points about this topic based EXCLUSIVELY on the sources above.
Format: Start each bullet with "- " (dash and space)
IMPORTANT: Output ONLY the bullet points, nothing else."""

        try:
            response = gemini_model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=800,
                ),
            )

            response_text = getattr(response, "text", "").strip()

            if response_text:
                response_text = self._clean_rag_response(response_text)
                citation_count = len(re.findall(r'\[\d+\]', response_text))

                if citation_count >= 3:
                    confidence = "high"
                elif citation_count >= 1:
                    confidence = "medium"
                else:
                    confidence = "low"

                return {
                    "response": response_text,
                    "sources": [
                        {
                            "source": m.get("source", "Unknown"),
                            "category": m.get("category", "general"),
                            "topic": m.get("topic", "General"),
                        }
                        for m in metadatas
                    ],
                    "confidence": confidence,
                    "retrieved_docs": len(documents),
                }

            logger.warning("Gemini returned empty content.")

        except Exception as e:
            logger.error(f"RAG generation error: {e}")

        return self._create_fallback_response(metadatas, len(documents))

    def _clean_rag_response(self, text: str) -> str:
        """Clean and format RAG response."""
        text = re.sub(r'```[\w]*\n?', '', text)
        text = re.sub(r'```', '', text)

        lines = text.split('\n')
        bullet_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                line = re.sub(r'^[•*]\s*', '- ', line)
                bullet_lines.append(line)
            elif line and bullet_lines:
                bullet_lines[-1] += ' ' + line

        cleaned = '\n'.join(bullet_lines)
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)

        return cleaned.strip()

    def _create_fallback_response(self, metadatas: List[Dict], doc_count: int) -> Dict:
        """Create safe fallback response when Gemini fails."""
        fallback_bullets = []

        for i, meta in enumerate(metadatas[:5], 1):
            topic = meta.get("topic", "Clinical consideration")
            source = meta.get("source", "Clinical guidelines")
            fallback_bullets.append(
                f"- Evidence from {source} addresses {topic.lower()} in older adults [{i}]"
            )

        if not fallback_bullets:
            fallback_bullets.append(
                "- Limited specific evidence available in knowledge base for this query"
            )

        return {
            "response": '\n'.join(fallback_bullets),
            "sources": [
                {
                    "source": m.get("source", "Unknown"),
                    "category": m.get("category", "general"),
                    "topic": m.get("topic", "General"),
                }
                for m in metadatas
            ],
            "confidence": "low",
            "retrieved_docs": doc_count,
        }

    def get_collection_stats(self) -> Dict:
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name,
            "persist_directory": self.persist_directory,
        }


# ================================================================
# KNOWLEDGE LOADER
# ================================================================

class MedicalKnowledgeLoader:
    """Hard-coded curated medical knowledge for RAG."""

    @staticmethod
    def load_beers_criteria() -> List[Dict]:
        return [
            {
                "text": (
                    "Benzodiazepines (e.g., diazepam, lorazepam, alprazolam) in older adults are "
                    "linked to increased falls, fractures, cognitive impairment and delirium."
                ),
                "source": "AGS Beers Criteria 2023",
                "category": "Beers_PIM",
                "priority": 1,
                "topic": "Benzodiazepines – falls & cognition",
            },
            {
                "text": (
                    "First-generation antihistamines such as diphenhydramine and hydroxyzine have "
                    "strong anticholinergic and sedative effects, causing confusion and falls."
                ),
                "source": "AGS Beers Criteria 2023",
                "category": "Beers_PIM",
                "priority": 1,
                "topic": "Anticholinergic antihistamines",
            },
            {
                "text": (
                    "Tricyclic antidepressants (amitriptyline, doxepin) are highly anticholinergic "
                    "and may cause orthostatic hypotension, sedation and falls in older adults."
                ),
                "source": "AGS Beers Criteria 2023",
                "category": "Beers_PIM",
                "priority": 1,
                "topic": "TCAs – anticholinergic burden",
            },
            {
                "text": (
                    "Chronic NSAID use in older adults increases the risk of gastrointestinal "
                    "bleeding and kidney injury. Use is typically time-limited and monitored."
                ),
                "source": "AGS Beers Criteria 2023",
                "category": "Beers_PIM",
                "priority": 2,
                "topic": "NSAIDs – GI and renal risk",
            },
            {
                "text": (
                    "Long-term proton pump inhibitor therapy beyond about 8 weeks in older adults "
                    "has been associated with bone loss, fractures and C. difficile infection."
                ),
                "source": "AGS Beers Criteria 2023",
                "category": "Beers_PIM",
                "priority": 2,
                "topic": "PPIs – long-term safety",
            },
        ]

    @staticmethod
    def load_ddi_knowledge() -> List[Dict]:
        return [
            {
                "text": (
                    "Warfarin used together with aspirin has been associated with higher bleeding "
                    "risk, including gastrointestinal and intracranial hemorrhage in older adults."
                ),
                "source": "DrugBank Anticoagulation References",
                "category": "DDI",
                "priority": 1,
                "topic": "Warfarin + aspirin interaction",
            },
            {
                "text": (
                    "ACE inhibitors or ARBs used with spironolactone or other potassium-sparing "
                    "diuretics may lead to hyperkalemia, particularly in reduced kidney function."
                ),
                "source": "DrugBank DDI Summary",
                "category": "DDI",
                "priority": 1,
                "topic": "ACEi/ARB + K-sparing diuretic",
            },
            {
                "text": (
                    "Digoxin levels may rise when combined with amiodarone due to reduced "
                    "digoxin clearance, increasing risk of toxicity in older adults."
                ),
                "source": "DrugBank DDI Summary",
                "category": "DDI",
                "priority": 1,
                "topic": "Digoxin + amiodarone",
            },
            {
                "text": (
                    "SSRIs used together with tramadol or other serotonergic agents may be "
                    "associated with serotonin syndrome."
                ),
                "source": "DrugBank DDI Summary",
                "category": "DDI",
                "priority": 1,
                "topic": "SSRIs + tramadol",
            },
        ]

    @staticmethod
    def load_renal_dosing() -> List[Dict]:
        return [
            {
                "text": (
                    "Gabapentin is cleared primarily by the kidneys. In reduced renal function, "
                    "accumulation may lead to excessive sedation or dizziness."
                ),
                "source": "FDA Prescribing Information",
                "category": "Renal_Dosing",
                "priority": 1,
                "topic": "Gabapentin renal adjustment",
            },
            {
                "text": (
                    "Metformin is generally not used when kidney function is severely reduced "
                    "because of concern for lactic acidosis."
                ),
                "source": "FDA Prescribing Information",
                "category": "Renal_Dosing",
                "priority": 1,
                "topic": "Metformin eGFR limits",
            },
            {
                "text": (
                    "Dabigatran has substantial renal clearance. In low creatinine clearance, "
                    "dose exposure may rise and bleeding risk can increase."
                ),
                "source": "FDA Prescribing Information",
                "category": "Renal_Dosing",
                "priority": 1,
                "topic": "Dabigatran renal clearance",
            },
        ]

    @staticmethod
    def load_clinical_guidelines() -> List[Dict]:
        return [
            {
                "text": (
                    "Anticoagulation in older adults with atrial fibrillation involves balancing "
                    "stroke prevention and bleeding risk. Direct oral anticoagulants are often "
                    "considered when kidney function is adequate."
                ),
                "source": "ACC/AHA Guidelines 2023",
                "category": "Clinical_Guidelines",
                "priority": 1,
                "topic": "Anticoagulation in elderly",
            },
            {
                "text": (
                    "Medications that may increase fall risk in older adults include sedative-"
                    "hypnotics, antipsychotics, certain antidepressants, anticonvulsants and "
                    "drugs causing orthostatic hypotension."
                ),
                "source": "AGS Fall Prevention Guidance",
                "category": "Clinical_Guidelines",
                "priority": 1,
                "topic": "Fall-risk medications",
            },
        ]

    @staticmethod
    def load_all_knowledge() -> List[Dict]:
        all_docs = []
        all_docs.extend(MedicalKnowledgeLoader.load_beers_criteria())
        all_docs.extend(MedicalKnowledgeLoader.load_ddi_knowledge())
        all_docs.extend(MedicalKnowledgeLoader.load_renal_dosing())
        all_docs.extend(MedicalKnowledgeLoader.load_clinical_guidelines())
        logger.info(f"Prepared {len(all_docs)} curated medical documents.")
        return all_docs


# ================================================================
# INITIALIZATION FUNCTION - called by app.py only, NOT auto-run here
# ================================================================

def initialize_rag_system() -> MedicalRAG:
    """
    Create and populate the RAG system.
    Called explicitly by app.py - never runs automatically on import.
    """
    logger.info("Initializing Medical RAG System...")

    rag = MedicalRAG(reset=False)

    if rag.collection.count() == 0:
        logger.info("Knowledge base empty — ingesting documents...")
        knowledge_docs = MedicalKnowledgeLoader.load_all_knowledge()
        rag.ingest_documents(knowledge_docs)
    else:
        logger.info("Knowledge base already populated. Skipping ingestion.")

    stats = rag.get_collection_stats()
    logger.info(f"RAG System Ready: {stats}")

    return rag


# ================================================================
# IMPORTANT: No code runs automatically below this line.
# initialize_rag_system() is ONLY called when app.py imports and
# explicitly invokes it. Never put bare function calls here.
# ================================================================