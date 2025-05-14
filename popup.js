document.addEventListener("DOMContentLoaded", () => {
  console.log("✅ popup.js loaded");

  const checkBtn = document.getElementById("checkBtn");

  checkBtn.addEventListener("click", async () => {
    console.log("🔘 Button clicked");

    const subject = document.getElementById("subject").value.trim();
    const body = document.getElementById("body").value.trim();
    const resultDiv = document.getElementById("result");
    resultDiv.textContent = "";

    if (!subject || !body) {
      resultDiv.textContent = "❗ Please enter both subject and body.";
      return;
    }

    try {
      console.log("📤 Sending request...");
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ subject, body })
      });

      const result = await response.json();
      console.log("✅ Response received:", result);

      const message = `Result: ${result.label}\nConfidence: ${result.confidence}%`;
      resultDiv.textContent = message;

      // ✅ Send notification to service worker
      chrome.runtime.sendMessage({
        type: "notify",
        title: "Phishing Email Check",
        message: `${result.label} (Confidence: ${result.confidence}%)`
      });

    } catch (error) {
      console.error("❌ Fetch error:", error);
      resultDiv.textContent = "❌ Could not connect to the prediction server.";
    }
  });
});
