chrome.runtime.onInstalled.addListener(() => {
  chrome.notifications.create({
    type: "basic",
    iconUrl: "icon.png",
    title: "Test Notification",
    message: "Extension installed and notifications working!",
    priority: 2
  });
});
