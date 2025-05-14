chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "notify") {
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icon.png",
      title: message.title,
      message: message.message
    });
  }
});
