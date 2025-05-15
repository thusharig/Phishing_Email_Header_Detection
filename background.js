console.log("Phishing Detector background script loaded");

chrome.runtime.onInstalled.addListener(() => {
  console.log("Extension installed");

  chrome.notifications.create({
    type: "basic",
    iconUrl: "icon.png",
    title: "Phishing Detector Ready",
    message: "Extension installed and running.",
    priority: 2
  });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "showNotification") {
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icon.png",
      title: "Phishing Warning",
      message: message.text,
      priority: 2
    });
  }
});

chrome.action.onClicked.addListener(async () => {
  console.log("Extension icon clicked");

  const subject = "URGENT: Reset your password";
  const body = "Please click http://phish.me to update your password.";
  const sender = "fake@bank.ru";
  const date = new Date().toISOString();
  const urls = 1;

  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ subject, body, sender, date, urls })
    });

    const result = await response.json();

    chrome.notifications.create({
      type: "basic",
      iconUrl: "icon.png",
      title: `Prediction: ${result.label}`,
      message: `Confidence: ${result.confidence}%`,
      priority: 2
    });

  } catch (error) {
    console.error("API Error:", error);

    chrome.notifications.create({
      type: "basic",
      iconUrl: "icon.png",
      title: "Prediction Failed",
      message: "Could not connect to the phishing detection server.",
      priority: 2
    });
  }
});
