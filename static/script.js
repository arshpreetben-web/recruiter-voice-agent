let recordBtn = document.getElementById("recordBtn");
let status = document.getElementById("status");
let candidateText = document.getElementById("candidateText");
let sentimentText = document.getElementById("sentimentText");

let uploadBtn = document.getElementById("uploadBtn");
let skillMatch = document.getElementById("skillMatch");

let mediaRecorder;
let audioChunks = [];

recordBtn.addEventListener("click", async () => {
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = e => {
            audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            audioChunks = [];
            const formData = new FormData();
            formData.append("audio", audioBlob, "response.wav");

            status.innerText = "Processing...";

            try {
        const res = await fetch("/voice_input", {
            method: "POST",
            body: formData
        });

        // Try parsing safely
        const data = await res.json();

        candidateText.innerText = "Candidate said: " + data.text;
        sentimentText.innerText = `Tone: ${data.sentiment.label}, Confidence: ${data.sentiment.score.toFixed(2)}`;
        status.innerText = "Click to record again";
    } catch (err) {
        console.error("Error:", err);
        status.innerText = "Error processing audio.";
    }
};

        mediaRecorder.start();
        status.innerText = "Recording... Click again to stop.";
    } else {
        mediaRecorder.stop();
    }
});

// Resume & JD Upload
uploadBtn.addEventListener("click", async () => {
    const resumeFile = document.getElementById("resumeFile").files[0];
    const jdFile = document.getElementById("jdFile").files[0];

    if (!resumeFile || !jdFile) {
        skillMatch.innerText = "Please upload both Resume and JD.";
        return;
    }

    const formData = new FormData();
    formData.append("resume", resumeFile);
    formData.append("jd", jdFile);

    skillMatch.innerText = "Analyzing...";

    const res = await fetch("/analyze_resume", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    skillMatch.innerText = `Match: ${data.match_percentage}% | Skills Matched: ${data.skills_matched.join(", ")}`;
});
fetch("http://127.0.0.1:5000/voice_input", {
  method: "POST",
  body: formData
})
.then(async response => {
  const data = await response.json().catch(() => null);

  if (!response.ok || !data) {
    console.error("❌ Server returned an error:", data);
    alert("⚠️ Server error — check Flask console for details.");
    return;
  }

  console.log("✅ Server Response:", data);

  // 🧠 Fix: check if sentiment exists before accessing label
  const sentimentLabel = data.sentiment?.label || "UNKNOWN";
  const responseText = data.response || data.error || "No response";

  // Display result safely
  document.getElementById("response").innerText =
    `${responseText} (Tone: ${sentimentLabel})`;
})
.catch(err => {
  console.error("❌ Fetch error:", err);
  alert("Network error — Flask might have crashed or restarted.");
});

