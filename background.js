chrome.runtime.onInstalled.addListener(() => {
  console.log("Phishing Detector Extension Installed");
});

chrome.action.onClicked.addListener(async (tab) => {
  const subject = "URGENT: Reset your password";
  const body = "Please click http://phish.me to update your password.";
  const sender = "fake@bank.ru";
  const date = new Date().toISOString();
  const urls = 1;

  const response = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      subject,
      body,
      sender,
      date,
      urls
    })
  });

  const result = await response.json();

  chrome.notifications.create({
    type: "basic",
    iconUrl: "icon.png",
    title: `Email ${result.label}`,
    message: `Confidence: ${result.confidence}%`,
    priority: 2
  });
});
