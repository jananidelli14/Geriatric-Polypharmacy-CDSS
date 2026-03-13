const saveBtn = document.getElementById("saveBtn");
const minutesInput = document.getElementById("minutes");
const notesInput = document.getElementById("notes");
const historyList = document.getElementById("historyList");

const checkboxes = {
  aptitude: document.getElementById("aptitude"),
  coding: document.getElementById("coding"),
  core: document.getElementById("core"),
  revision: document.getElementById("revision")
};

const levelTitles = [
  "Beginner", "Learning", "Consistent", "Dedicated", "Expert",
  "Master", "Legend", "Champion", "Elite", "Genius"
];

function getToday() {
  return new Date().toISOString().split("T")[0];
}

function getWeekStart() {
  const now = new Date();
  const dayOfWeek = now.getDay();
  const diff = dayOfWeek === 0 ? 6 : dayOfWeek - 1;
  const weekStart = new Date(now);
  weekStart.setDate(now.getDate() - diff);
  weekStart.setHours(0, 0, 0, 0);
  return weekStart;
}

let data = {
  lastDate: null,
  currentStreak: 0,
  longestStreak: 0,
  totalDays: 0,
  totalMinutes: 0,
  xp: 0,
  level: 1,
  history: []
};

function loadData() {
  try {
    const stored = localStorage.getItem("habitData");
    if (stored) {
      data = { ...data, ...JSON.parse(stored) };
    }
  } catch (e) {
    console.error("Error loading data:", e);
  }
}

function saveData() {
  try {
    localStorage.setItem("habitData", JSON.stringify(data));
  } catch (e) {
    console.error("Error saving data:", e);
  }
}

function calculateLevel(xp) {
  return Math.floor(xp / 100) + 1;
}

function getXPForNextLevel(level) {
  return level * 100;
}

function updateLevelUI() {
  const level = calculateLevel(data.xp);
  const xpInCurrentLevel = data.xp % 100;
  const xpForNextLevel = 100;
  const progress = (xpInCurrentLevel / xpForNextLevel) * 100;

  document.getElementById("userLevel").textContent = level;
  document.getElementById("levelTitle").textContent = levelTitles[Math.min(level - 1, levelTitles.length - 1)];
  document.getElementById("currentXP").textContent = xpInCurrentLevel;
  document.getElementById("nextLevelXP").textContent = xpForNextLevel;
  document.getElementById("xpProgress").style.width = progress + "%";

  // Update badge locks
  const badges = document.querySelectorAll(".reward-badge");
  badges.forEach((badge, index) => {
    if (level > index * 2) {
      badge.classList.remove("locked");
    } else {
      badge.classList.add("locked");
    }
  });
}

function calculateWeekHours() {
  const weekStart = getWeekStart();
  let weekMinutes = 0;
  
  data.history.forEach(entry => {
    const entryDate = new Date(entry.date);
    if (entryDate >= weekStart) {
      weekMinutes += entry.minutes;
    }
  });
  
  return Math.round(weekMinutes / 60);
}

function updateUI() {
  document.getElementById("currentStreak").textContent = data.currentStreak;
  document.getElementById("longestStreak").textContent = data.longestStreak;
  document.getElementById("totalHours").textContent = Math.round(data.totalMinutes / 60);
  document.getElementById("weekHours").textContent = calculateWeekHours();
  updateLevelUI();
  updateHistory();
}

function updateHistory() {
  if (data.history.length === 0) {
    historyList.innerHTML = '<div class="empty-state">No study sessions yet. Start tracking today!</div>';
    return;
  }

  historyList.innerHTML = data.history.slice(-10).reverse().map(entry => `
    <div class="history-item">
      <div class="history-date">${new Date(entry.date).toLocaleDateString('en-US', { 
        weekday: 'short', 
        month: 'short', 
        day: 'numeric' 
      })}</div>
      <div class="history-details">⏱️ ${entry.minutes} minutes${entry.xpGained ? ` • +${entry.xpGained} XP` : ''}</div>
      ${entry.topics.length > 0 ? `
        <div class="history-topics">
          ${entry.topics.map(t => `<span class="topic-badge">${t}</span>`).join('')}
        </div>
      ` : ''}
      ${entry.notes ? `<div class="history-details" style="margin-top: 8px;">${entry.notes}</div>` : ''}
    </div>
  `).join('');
}

function showCelebration(type, message) {
  const modal = document.getElementById("celebrationModal");
  const title = document.getElementById("celebrationTitle");
  const text = document.getElementById("celebrationText");
  
  if (type === "levelup") {
    title.textContent = "🎉 Level Up!";
  } else if (type === "streak") {
    title.textContent = "🔥 Streak Milestone!";
  }
  
  text.textContent = message;
  modal.style.display = "block";
  
  setTimeout(() => {
    modal.style.display = "none";
  }, 3000);
}

function closeCelebration() {
  document.getElementById("celebrationModal").style.display = "none";
}

saveBtn.addEventListener("click", () => {
  const today = getToday();
  const minutes = Number(minutesInput.value);
  const notes = notesInput.value.trim();

  if (minutes <= 0) {
    alert("⚠️ Please enter the minutes you studied today");
    return;
  }

  const topics = [];
  if (checkboxes.aptitude.checked) topics.push("Aptitude");
  if (checkboxes.coding.checked) topics.push("Coding");
  if (checkboxes.core.checked) topics.push("Core");
  if (checkboxes.revision.checked) topics.push("Revision");

  if (data.lastDate && data.lastDate !== today) {
    const last = new Date(data.lastDate);
    const now = new Date(today);
    const diff = Math.floor((now - last) / (1000 * 60 * 60 * 24));

    if (diff === 1) {
      data.currentStreak += 1;
      if (data.currentStreak % 7 === 0) {
        showCelebration("streak", `Amazing! ${data.currentStreak} day streak! 🔥`);
      }
    } else if (diff > 1) {
      data.currentStreak = 1;
    }
  } else if (!data.lastDate) {
    data.currentStreak = 1;
  } else if (data.lastDate === today) {
    alert("✅ You've already logged today! Keep it up tomorrow.");
    return;
  }

  const oldLevel = calculateLevel(data.xp);
  const xpGained = Math.floor(minutes / 10) + (topics.length * 5);
  data.xp += xpGained;
  const newLevel = calculateLevel(data.xp);

  data.lastDate = today;
  data.totalDays += 1;
  data.totalMinutes += minutes;

  if (data.currentStreak > data.longestStreak) {
    data.longestStreak = data.currentStreak;
  }

  data.history.push({
    date: today,
    minutes,
    topics,
    notes,
    xpGained
  });

  saveData();
  updateUI();

  if (newLevel > oldLevel) {
    showCelebration("levelup", `You're now Level ${newLevel} - ${levelTitles[Math.min(newLevel - 1, levelTitles.length - 1)]}!`);
  }

  minutesInput.value = "";
  notesInput.value = "";
  Object.values(checkboxes).forEach(cb => cb.checked = false);

  if (newLevel <= oldLevel) {
    alert(`🎉 +${xpGained} XP! ${data.currentStreak} day streak!`);
  }
});

loadData();
updateUI();