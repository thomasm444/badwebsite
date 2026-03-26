const gatePanel = document.getElementById("gatePanel");
const cameraPanel = document.getElementById("cameraPanel");
const gateMessage = document.getElementById("gateMessage");
const roastOutput = document.getElementById("roastOutput");
const promptExpression = document.getElementById("promptExpression");
const modelStatus = document.getElementById("modelStatus");

const obviousBtn = document.getElementById("obviousBtn");
const maybeBtn = document.getElementById("maybeBtn");
const trapButtons = document.querySelectorAll(".trap");
const runnerBtn = document.getElementById("runnerBtn");
const chaosZone = document.getElementById("chaosZone");
const consentCheck = document.getElementById("consentCheck");
const finalCameraBtn = document.getElementById("finalCameraBtn");

const retryBtn = document.getElementById("retryBtn");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const stepEls = [...document.querySelectorAll(".gate-step")];
let currentStep = 1;
let stream = null;
let faceLandmarker = null;
let expressionModelReady = false;
let expressionModelLoading = false;
let runnerMovesLeft = 15;
let runnerHoverCount = 0;
const requiredRunnerHovers = 15;
let currentChallenge = null;
let autoAnalyzeTimer = null;
let isAnalyzing = false;
let challengeResolved = false;
let challengePickCount = 0;

const expressionChallenges = [
  {
    key: "smile",
    label: "Smile",
    emoji: "🙂",
    comments: [
      "That's the saddest smile I've ever processed",
      "My error messages are more cheerful than that",
      "Did it hurt? Smiling clearly doesn't come natural to you",
      "You smile like you're filling out tax forms"
    ]
  },
  {
    key: "neutral",
    label: "Neutral",
    emoji: "😐",
    comments: [
      "Congratulations, you look like a default avatar",
      "Absolutely nothing is happening on your face. Impressive.",
      "You have the energy of an unread notification",
      "A blank wall has more personality than this"
    ]
  },
  {
    key: "confused",
    label: "Confused",
    emoji: "😕",
    comments: [
      "Your face buffered and never loaded",
      "Even the camera doesn't know what it's looking at",
      "You look confused AND concerning. Impressive multitasking",
      "This is your thinking face? Yikes."
    ]
  },
  {
    key: "sad-face",
    label: "Sad",
    emoji: "😢",
    comments: [
      "The website feels nothing. Unlike you apparently.",
      "Bold of you to cry in front of a website",
      "Error 404: Sympathy not found",
      "Even your sadness is mid"
    ]
  }
];

function randomFrom(list) {
  return list[Math.floor(Math.random() * list.length)];
}

function speakRoast(text) {
  if (!("speechSynthesis" in window)) {
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  utterance.volume = 1;
  window.speechSynthesis.speak(utterance);
}

function pickChallenge() {
  const easyChallenges = expressionChallenges.filter(
    (item) => item.key === "smile" || item.key === "neutral"
  );

  if (challengePickCount < 4 && easyChallenges.length) {
    currentChallenge = randomFrom(easyChallenges);
  } else {
    currentChallenge = randomFrom(expressionChallenges);
  }

  challengePickCount += 1;
  challengeResolved = false;
  promptExpression.textContent = `Make this face: ${currentChallenge.emoji} ${currentChallenge.label}`;
}

function startAutoAnalyze() {
  if (autoAnalyzeTimer) {
    clearInterval(autoAnalyzeTimer);
  }

  autoAnalyzeTimer = setInterval(() => {
    roastCurrentFrame({ auto: true, silentMismatch: true });
  }, 1700);

  roastCurrentFrame({ auto: true, silentMismatch: true });
}

async function ensureExpressionModel() {
  if (expressionModelReady || expressionModelLoading) {
    return;
  }

  expressionModelLoading = true;
  modelStatus.textContent = "Loading expression detector...";

  try {
    const vision = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3");
    const fileset = await vision.FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );

    faceLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
      },
      outputFaceBlendshapes: true,
      runningMode: "IMAGE",
      numFaces: 1
    });

    expressionModelReady = true;
    modelStatus.textContent = "Expression detector ready.";
  } catch (err) {
    modelStatus.textContent = "Expression detector failed to load. Check internet and retry.";
    console.error(err);
  } finally {
    expressionModelLoading = false;
  }
}

function getBlendshapeScores(result) {
  const scoreMap = {};
  const categories = result?.faceBlendshapes?.[0]?.categories || [];
  categories.forEach((entry) => {
    scoreMap[entry.categoryName] = entry.score;
  });
  return scoreMap;
}

function inferExpression(scores) {
  const smile = Math.max(scores.mouthSmileLeft || 0, scores.mouthSmileRight || 0);
  const frown = Math.max(scores.mouthFrownLeft || 0, scores.mouthFrownRight || 0);
  const browInnerUp = scores.browInnerUp || 0;
  const eyeWide = ((scores.eyeWideLeft || 0) + (scores.eyeWideRight || 0)) / 2;
  const eyeSquint = ((scores.eyeSquintLeft || 0) + (scores.eyeSquintRight || 0)) / 2;
  const mouthPucker = scores.mouthPucker || 0;
  const mouthShrugUpper = scores.mouthShrugUpper || 0;

  const sadFaceScore = frown + browInnerUp + eyeSquint;
  const confusedScore = browInnerUp + eyeWide + mouthPucker + mouthShrugUpper;

  if (smile > 0.35) {
    return expressionChallenges.find((item) => item.key === "smile");
  }

  if (sadFaceScore > 0.75) {
    return expressionChallenges.find((item) => item.key === "sad-face");
  }

  if (confusedScore > 0.95) {
    return expressionChallenges.find((item) => item.key === "confused");
  }

  return expressionChallenges.find((item) => item.key === "neutral");
}

function showStep(stepNumber) {
  currentStep = stepNumber;
  stepEls.forEach((el) => {
    const shouldShow = Number(el.dataset.step) === stepNumber;
    el.classList.toggle("hidden", !shouldShow);
  });
}

function nudgeRunnerButton() {
  runnerHoverCount += 1;

  if (runnerMovesLeft <= 0) {
    gateMessage.textContent = `Hover count: ${Math.min(runnerHoverCount, requiredRunnerHovers)}/${requiredRunnerHovers}. You can click now.`;
    return;
  }

  const zoneRect = chaosZone.getBoundingClientRect();
  const btnRect = runnerBtn.getBoundingClientRect();

  const maxX = Math.max(5, zoneRect.width - btnRect.width - 8);
  const maxY = Math.max(5, zoneRect.height - btnRect.height - 8);

  const x = Math.floor(Math.random() * maxX);
  const y = Math.floor(Math.random() * maxY);

  runnerBtn.style.left = `${x}px`;
  runnerBtn.style.top = `${y}px`;
  runnerMovesLeft -= 1;

  if (runnerHoverCount < requiredRunnerHovers) {
    gateMessage.textContent = `Hover count: ${runnerHoverCount}/${requiredRunnerHovers}. Keep chasing.`;
  } else {
    gateMessage.textContent = `Hover count: ${requiredRunnerHovers}/${requiredRunnerHovers}. Click is unlocked.`;
  }
}

obviousBtn.addEventListener("click", () => {
  gateMessage.textContent = "Great, one click down. Three annoyances to go.";
  showStep(2);
});

trapButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    gateMessage.textContent = "Wrong. Those were decoys. Try not to panic.";
    showStep(Number(btn.dataset.next));
  });
});

maybeBtn.addEventListener("click", () => {
  gateMessage.textContent = "Correct-ish. Proceed to chaos mode and hover 15 times.";
  showStep(3);
  runnerMovesLeft = 15;
  runnerHoverCount = 0;
});

runnerBtn.addEventListener("mouseenter", nudgeRunnerButton);
runnerBtn.addEventListener("mousemove", nudgeRunnerButton);
runnerBtn.addEventListener("click", () => {
  if (runnerHoverCount < requiredRunnerHovers) {
    gateMessage.textContent = `Nope. Hover ${requiredRunnerHovers - runnerHoverCount} more time(s) first.`;
    return;
  }

  gateMessage.textContent = "Impressive. You clicked the greased button.";
  showStep(4);
});

consentCheck.addEventListener("change", () => {
  finalCameraBtn.disabled = !consentCheck.checked;
});

finalCameraBtn.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }
    });

    video.srcObject = stream;
    gatePanel.classList.add("hidden");
    cameraPanel.classList.remove("hidden");
    await ensureExpressionModel();
    pickChallenge();
    startAutoAnalyze();
  } catch (err) {
    gateMessage.textContent =
      "Camera blocked. If you denied permission, refresh and suffer through the gate again.";
    console.error(err);
  }
});

async function roastCurrentFrame(options = {}) {
  const { auto = false, silentMismatch = false } = options;

  if (isAnalyzing) {
    return;
  }

  if (auto && challengeResolved) {
    return;
  }

  if (!stream || video.readyState < 2) {
    if (!auto) {
      roastOutput.textContent = "Camera is not ready. Hold your dramatic face for a second.";
    }
    return;
  }

  isAnalyzing = true;

  try {
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (!expressionModelReady || !faceLandmarker) {
      if (!auto) {
        roastOutput.textContent = "Expression detector is still loading. Try again in a second.";
      }
      return;
    }

    const result = faceLandmarker.detect(canvas);

    if (!result?.faceBlendshapes?.length) {
      if (!auto) {
        roastOutput.textContent = "I asked for a face and got absolutely nothing. Try again.";
      }
      return;
    }

    const detectedExpression = inferExpression(getBlendshapeScores(result));

    if (!currentChallenge) {
      pickChallenge();
    }

    if (detectedExpression.key !== currentChallenge.key) {
      if (!silentMismatch) {
        roastOutput.textContent = `Detected: ${detectedExpression.emoji} ${detectedExpression.label}. Requested: ${currentChallenge.emoji} ${currentChallenge.label}. Try again.`;
      }
      return;
    }

    const chosenRoast = randomFrom(currentChallenge.comments);
    roastOutput.textContent = chosenRoast;
    speakRoast(chosenRoast);
    challengeResolved = true;
  } catch (err) {
    if (!auto) {
      roastOutput.textContent = "Your face confused the detector. Try one more time.";
    }
    console.error(err);
  } finally {
    isAnalyzing = false;
  }
}

retryBtn.addEventListener("click", () => {
  pickChallenge();
  roastOutput.textContent = "New round. Match the requested emoji expression.";
});

showStep(1);
