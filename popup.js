document.addEventListener("DOMContentLoaded", () => {
  const analyzeBtn = document.getElementById("analyzeBtn");
  const headerInput = document.getElementById("headerInput");
  const resultDiv = document.getElementById("result");

  analyzeBtn.addEventListener("click", () => {
    const headers = headerInput.value.trim();
    const findings = [];

    // SPF Check
    if (headers.match(/Received-SPF:\s*pass/i)) {
      findings.push("✅ SPF Passed");
    } else {
      findings.push("⚠️ SPF check failed or missing");
    }

    // DKIM Check
    if (headers.match(/dkim=pass/i)) {
      findings.push("✅ DKIM Passed");
    } else {
      findings.push("⚠️ DKIM check failed or missing");
    }

    // From and Reply-To mismatch
    const fromMatch = headers.match(/From:\s(.+)/i);
    const replyToMatch = headers.match(/Reply-To:\s(.+)/i);
    if (fromMatch && replyToMatch && fromMatch[1].trim() !== replyToMatch[1].trim()) {
      findings.push("⚠️ 'Reply-To' differs from 'From' — possible spoofing.");
    } else {
      findings.push("✅ From and Reply-To match");
    }

    // Return-Path check
    const returnPathMatch = headers.match(/Return-Path:\s<([^>]+)>/i);
    if (returnPathMatch && fromMatch && !fromMatch[1].includes(returnPathMatch[1])) {
      findings.push("⚠️ Return-Path differs from sender — possible spoofing.");
    }

    resultDiv.innerHTML = findings.map(f => `<p>${f}</p>`).join("");

    // Show notification if any warning found
    if (findings.some(f => f.includes("⚠️"))) {
      chrome.runtime.sendMessage({
        type: "showNotification",
        text: "⚠️ Potential phishing detected. Review the email header carefully."
      });
    }
  });
});
