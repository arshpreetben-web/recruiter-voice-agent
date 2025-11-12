// 🎙️ AI Mock Interview Assistant
console.log("✅ script.js loaded successfully");

// 🌟 Element references
const recordBtn = document.getElementById("recordBtn");
const status = document.getElementById("status");
const candidateText = document.getElementById("candidateText");
const sentimentText = document.getElementById("sentimentText");
const uploadBtn = document.getElementById("uploadBtn");
const skillMatch = document.getElementById("skillMatch");
let mediaRecorder, audioChunks = [];
let currentAudio = null;

// Add score display area dynamically
const scoreContainer = document.createElement("div");
scoreContainer.classList.add("mt-3");
scoreContainer.innerHTML = `
  <div class="mb-2"><b>📊 Answer Quality</b></div>
  <div class="mb-2">
    <small>🧠 Clarity:</small>
    <div class="progress" style="height: 10px;">
      <div id="clarityBar" class="progress-bar bg-info" style="width: 0%;"></div>
    </div>
    <small id="clarityScore" class="text-muted">0%</small>
  </div>
  <div class="mb-2">
    <small>💡 Content:</small>
    <div class="progress" style="height: 10px;">
      <div id="contentBar" class="progress-bar bg-success" style="width: 0%;"></div>
    </div>
    <small id="contentScore" class="text-muted">0%</small>
  </div>
`;
document.querySelector(".container").appendChild(scoreContainer);

// 🔔 Toast Notification
function showToast(message, type = "info") {
  const toastContainer = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast align-items-center text-white border-0 bg-${type}`;
  toast.role = "alert";
  toast.innerHTML = `
    <div class="d-flex">
      <div class="toast-body">${message}</div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
    </div>`;
  toastContainer.appendChild(toast);
  const bsToast = new bootstrap.Toast(toast, { delay: 2500 });
  bsToast.show();
  toast.addEventListener("hidden.bs.toast", () => toast.remove());
}

// 🎙️ Record Voice
recordBtn.addEventListener("click", async () => {
  try {
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        const formData = new FormData();
        formData.append("audio", audioBlob, "response.wav");

        recordBtn.classList.remove("recording");
        status.innerText = "⏳ Processing your answer...";
        showToast("⏳ Processing your answer...", "warning");

        try {
          const res = await fetch("/voice_input", { method: "POST", body: formData });
          const data = await res.json();

          if (data.error) {
            showToast("⚠️ Error processing audio.", "danger");
            status.innerText = "⚠️ Error processing audio.";
            return;
          }

          // 🗣️ Candidate’s recognized text
          candidateText.innerText = `🗣️ You said: ${data.text}`;
          sentimentText.innerText = `🧠 Tone: ${data.sentiment.label} (${(
            data.sentiment.score * 100
          ).toFixed(1)}%)`;

          // 💬 Show AI feedback
          if (data.feedback) {
            const feedbackBox = document.createElement("div");
            feedbackBox.classList.add("feedback-box", "p-3", "mt-3", "rounded");
            feedbackBox.style.background = "#f1f3f4";
            feedbackBox.innerHTML = `<b>💬 Feedback:</b> ${data.feedback}`;
            document.querySelector(".container").appendChild(feedbackBox);
            showToast("✅ Feedback received!", "success");
          }

          // 📊 Update clarity/content progress bars
          if (data.clarity_score !== undefined && data.content_score !== undefined) {
            updateScoreBars(data.clarity_score, data.content_score);
          }

          // 🔁 Move to next question
          if (!data.is_last && data.next_question) {
            showToast("💬 Great! Let's move to the next question...", "info");
            status.innerText = "🧑‍💼 Interviewer is speaking...";

            const transitions = [
              "Alright, thank you for your answer.",
              "Got it. Let's move to the next one.",
              "Okay, that makes sense. Here's another question.",
              "Nice response. Let's continue with the next one."
            ];
            const randomLine = transitions[Math.floor(Math.random() * transitions.length)];

            await speakQuestion(randomLine);
            await new Promise(resolve => setTimeout(resolve, 800));
            await speakQuestion(data.next_question);
          } else {
            showToast("🎉 Interview Complete!", "success");
            status.innerText = "🎉 Mock Interview Finished!";
            setTimeout(() => (window.location.href = "/summary"), 4000);
          }
        } catch (err) {
          console.error("❌ Fetch error:", err);
          showToast("❌ Server error while processing.", "danger");
        }
      };

      mediaRecorder.start();
      recordBtn.classList.add("recording");
      status.innerText = "🎙️ Recording... Click again to stop.";
      showToast("🎙️ Recording started...", "info");
    } else {
      mediaRecorder.stop();
      showToast("🎧 Recording stopped.", "secondary");
    }
  } catch (err) {
    console.error("Microphone error:", err);
    showToast("🎤 Please allow microphone access!", "danger");
  }
});

// 📊 Update clarity/content score progress bars
function updateScoreBars(clarity, content) {
  const clarityBar = document.getElementById("clarityBar");
  const contentBar = document.getElementById("contentBar");
  const clarityScore = document.getElementById("clarityScore");
  const contentScore = document.getElementById("contentScore");

  clarityBar.style.width = `${clarity}%`;
  contentBar.style.width = `${content}%`;
  clarityScore.innerText = `${clarity}%`;
  contentScore.innerText = `${content}%`;

  clarityBar.classList.remove("bg-info", "bg-success", "bg-danger");
  contentBar.classList.remove("bg-info", "bg-success", "bg-danger");

  if (clarity >= 75) clarityBar.classList.add("bg-success");
  else if (clarity >= 50) clarityBar.classList.add("bg-info");
  else clarityBar.classList.add("bg-danger");

  if (content >= 75) contentBar.classList.add("bg-success");
  else if (content >= 50) contentBar.classList.add("bg-info");
  else contentBar.classList.add("bg-danger");
}

// 📂 Resume + JD Upload → Analyze → Start Interview
uploadBtn.addEventListener("click", async () => {
  const resumeFile = document.getElementById("resumeFile").files[0];
  const jdFile = document.getElementById("jdFile").files[0];

  if (!resumeFile || !jdFile) {
    skillMatch.innerText = "⚠️ Please upload both Resume and JD.";
    showToast("⚠️ Please upload both files!", "warning");
    return;
  }

  const formData = new FormData();
  formData.append("resume", resumeFile);
  formData.append("jd", jdFile);

  uploadBtn.innerHTML = `<div class="spinner-border spinner-border-sm text-light me-2"></div>Analyzing...`;
  uploadBtn.disabled = true;

  try {
    const res = await fetch("/analyze_resume", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      skillMatch.innerText = "❌ Error analyzing resume.";
      showToast("❌ Resume analysis failed.", "danger");
      return;
    }

    skillMatch.innerHTML = `
      ✅ <b>Match Score:</b> ${data.match_percentage}%<br>
      🧩 <b>Skills Matched:</b> ${data.skills_matched.join(", ") || "None"}
    `;
    showToast("✅ Resume analyzed successfully!", "success");

    await startInterview();
  } catch (err) {
    console.error("Error:", err);
    showToast("⚠️ Server error. Try again later.", "danger");
  } finally {
    uploadBtn.innerHTML = "🧠 Analyze & Start Interview";
    uploadBtn.disabled = false;
  }
});

// 🚀 Start Interview
async function startInterview() {
  const resumeFile = document.getElementById("resumeFile").files[0];
  const jdFile = document.getElementById("jdFile").files[0];
  const formData = new FormData();
  formData.append("resume", resumeFile);
  formData.append("jd", jdFile);

  try {
    const res = await fetch("/start_interview", { method: "POST", body: formData });
    const data = await res.json();
    console.log("🚀 Interview Started:", data);

    if (data.error) {
      showToast("❌ Error starting interview: " + data.error, "danger");
      return;
    }

    showToast("🎯 Interview started!", "success");
    await speakQuestion(data.first_question);
  } catch (err) {
    console.error("Error starting interview:", err);
    showToast("❌ Network error starting interview", "danger");
  }
}

// 🔊 Speak Question (Stable version)
async function speakQuestion(text) {
  try {
    if (!text.trim()) return;

    let interviewerBox = document.getElementById("interviewerText");
    if (!interviewerBox) {
      interviewerBox = document.createElement("p");
      interviewerBox.id = "interviewerText";
      interviewerBox.classList.add("interviewer-box", "p-2", "rounded", "mt-3");
      document.querySelector(".container").prepend(interviewerBox);
    }

    interviewerBox.innerText = `🧑‍💼 Interviewer: ${text}`;
    status.innerText = "🧑‍💼 Interviewer is speaking...";

    const formData = new FormData();
    formData.append("question", text);
    const res = await fetch("/speak_question", { method: "POST", body: formData });
    if (!res.ok) throw new Error("TTS fetch failed");

    const blob = await res.blob();
    const audioUrl = URL.createObjectURL(blob);

    if (currentAudio) {
      currentAudio.pause();
      currentAudio.currentTime = 0;
    }

    currentAudio = new Audio(audioUrl);
    currentAudio.volume = 1.0;

    await currentAudio.play();

    currentAudio.onended = () => {
      status.innerText = "🎙️ Your turn! Click record to answer.";
    };
  } catch (err) {
    console.error("❌ speakQuestion error:", err);
    showToast("⚠️ Could not play audio.", "warning");
  }
}
