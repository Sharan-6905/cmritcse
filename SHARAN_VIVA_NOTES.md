# Sharan: core explanation notes (viva and live demo)

Project: **Waste Classification and Recycling Optimization Using Agentic AI and Computer Vision**
Your part: the model, the agent, the optimizer, the numbers and the limits. Sharath covers the dataset and results; Yashwanth covers the app, Docker, GitHub and the live link.

---

## 1. The 30-second pitch (say this first)

"People mis-sort waste, and dirty or hazardous items spoil recycling. Our app takes a photo, finds each item, identifies the material with a trained image model, lets an AI agent double-check it, and then tells you which bin it goes in, how to prepare it, and how much more gets recovered if you prepare it properly."

---

## 2. Speaking order (about 6 minutes)

| # | Topic | Show | Time |
|---|---|---|---|
| 1 | Why our first version was weak, and what we fixed | Notebook V1 vs `train_colab.ipynb` | 1 min |
| 2 | The model: transfer learning and training | Notebook, model and training sections | 1.5 min |
| 3 | Explainability: Grad-CAM | A heatmap card in the app | 30 s |
| 4 | What "agentic" means here, and the three tools | Live step trail | 1.5 min |
| 5 | Why the agent overrides the model | "Agent override" badge | 30 s |
| 6 | The optimizer | As-is vs after-prep bars | 1 min |
| 7 | Honest limits | Insights page | 30 s |

---

## 3. The explanations, in plain language

### 3.1 Why the first version (V1) was weak
- **What happened:** V1 picked its best model using the same photos it reported the score on, and those photos had been randomly flipped and recoloured. So 92.89% looked better than reality. There was also no separate test set.
- **Example:** grading a student on the same practice questions they studied from.
- **What we fixed (V2):** a hidden test set (15% of photos, touched only once at the end), no distortion on validation and test photos, a `trash` class, class weights, and a fixed random seed so the results can be reproduced.

### 3.2 The model: transfer learning
- We use **ResNet50**, an image network that has already learned shapes, edges and textures from millions of photos.
- We **replaced its final layer** with one that outputs our 6 classes, then **fine-tuned the whole network** on our 2,527 photos with a small learning rate (1e-4), so it adjusts gently and keeps what it knew.
- **Example:** hiring an experienced sorter who already recognises objects and training them for one week on recycling, instead of teaching a newborn from scratch.
- **Why not train from scratch:** 2.5k photos are far too few.

### 3.3 How the training works (the notebook's training cell)
- **Epoch:** one full pass over the training photos. We run up to 15.
- **Validation:** after every epoch the model is checked on photos it never learns from. We keep the epoch with the best validation accuracy.
- **Early stopping:** if validation does not improve for 5 epochs, we stop, because more training would only memorise.
- **Augmentation:** training photos are randomly cropped, flipped, rotated and recoloured each time, so it cannot memorise exact pictures.
- **Class weights:** rare classes (trash: 137 photos) count a little more.
- **Test:** the hidden test photos are scored once at the end: **93.4%** on the Colab run, 91.8% when re-run locally.
- **Example:** revising for an exam. Each epoch is a study round; the mock test (validation) shows progress; you stop when mock marks stop rising; the real exam (test) is seen once.

### 3.4 Grad-CAM (explainability)
- It colours the parts of the photo that pushed the model towards its answer. Technically it uses the gradients of the predicted class with respect to the last convolutional layer's feature maps, turns them into weights, and paints a heatmap.
- **Example:** asking the sorter to point at what they looked at to decide "this is glass".
- **In the app:** the small "Where the model looked" picture on each item card.

### 3.5 What "agentic" means here
- A normal chatbot only replies. An **agent** has a **toolbox** and decides which tool to use and in what order. Ours is Gemini with three tools:
  1. **`analyze_image`:** the eyes. Finds each item (Gemini vision), crops it, and classifies each crop with the trained ResNet50.
  2. **`get_disposal_guide`:** the reference book. Returns the bin, steps, breakdown time and what to avoid for one of 11 categories.
  3. **`optimize_recycling`:** the calculator. Runs the optimizer on all items.
- **How a tool call works:** the model does not run code. It sends back a request ("call `get_disposal_guide` with category glass"), our server runs the real function, and gives the result back to the model, which then writes the answer. This is called **function calling**.
- **Example:** a site manager who looks at the problem, asks the specialist (the classifier) for an opinion, double-checks it, consults the manual and runs the numbers before giving you the plan.
- **What you can point at in the app:** the live step trail. Each step is a real event sent from the server as it happens.

### 3.6 Why the agent overrides the classifier
- The classifier only knows **6** materials and can be **confidently wrong** (a battery called metal at 99%).
- The agent also looks at the photo. It overrides when the classifier's confidence is below **0.6**, when the item is outside those 6 classes (food, battery, electronics, clothing, hazardous), or when the answer is clearly wrong. The app shows an **"Agent override"** badge.
- **Example:** a second opinion from a senior doctor.
- **Measured:** outside the dataset the classifier alone scored **63%**, and the full system **84%**.

### 3.7 The optimizer (the "recycling optimization" in the title)
- It is plain code, not the language model, so the numbers cannot be invented.
- It compares two cases for the same items:
  - **As-is:** dirty recyclables are rejected at the plant.
  - **After prep:** lightly dirty items (food residue) are rinsed and recovered; a heavily soiled paper or cardboard item salvages its clean half; other heavily contaminated items are lost.
- Output: items grouped by bin, **recovered share** (as-is vs after prep), **CO₂ avoided** and the prep steps.
- **Example:** a greasy pizza box. Binned as-is, the grease spoils it. Tear off the clean lid and that part is recycled. The app shows, for example, "recovered 38% → 86%".
- **Be honest:** weights are typical values, CO₂ factors are rough, and the contamination rules are our own simple assumptions. They are **estimates**.

### 3.8 Why accuracy drops outside the dataset (93% → 63%)
- The model studied clean, single-item photos. Real photos are cluttered.
- **Example:** a student who only practised textbook diagrams meets a messy photo.
- The agent's second look (Gemini) wins most of it back. Say plainly that part of the credit goes to Gemini's vision; the classifier is one tool inside the agent.

### 3.9 Making it fast and reliable
- Each Gemini call can be slow or out of free quota, so every call has a **20-second limit**, and the app falls back through **4 models** automatically.
- Photos are shrunk to 1024 px before upload, tool calls run together, and the written answer is short.
- Each visitor is limited to 40 requests an hour and the whole site to 250 a day, so strangers cannot use up the quota.
- **Example:** a restaurant with backup suppliers.

---

## 4. Numbers to remember

| Fact | Number |
|---|---|
| Training photos | **2,527** (6 classes) |
| Split | **70 / 15 / 15** = 1,768 / 379 / 380 |
| Test accuracy | **93.4%** (Colab), 91.8% (local re-run) |
| Weakest class | trash (137 photos, 20 test photos) |
| Outside photos: classifier alone | **63%** (12 of 19) |
| Outside photos: agent, same photos | **84%** (16 of 19) |
| Outside photos: agent, all | **83%** (24 of 29) |
| Hazardous items | 5 of 5 found (a smoke test, not an accuracy figure) |
| Categories / tools | **11** / **3** |
| Speed | about 11 to 35 s per photo (was 83 s) |

---

## 5. Live demo talk track

| Step | You do | You say |
|---|---|---|
| 1 | Upload one single item | "The photo goes to our trained ResNet50. Watch the steps." |
| 2 | Point at the card | "Here is its guess and confidence, and the heatmap shows where it looked." |
| 3 | Click an "Ask next" chip | "It remembers the conversation and suggests new questions." |
| 4 | Upload the multi-item photo | "Now several items. It finds each one, classifies each, then plans for all." |
| 5 | Point at the badge and bars | "The agent overrode the classifier here. And this shows recovery as-is versus after preparing the items." |
| 6 | Open Insights | "These are our real results, including where it is weaker." |

While an answer loads (15 to 40 s), **talk through the step trail**. Do not re-upload. If it is slow, say: "Our free AI quota is limited, so it can take a moment; the app switches models automatically."

---

## 6. Questions you are likely to get (and short answers)

| Question | Answer |
|---|---|
| What is ResNet50? | A 50-layer image network. Its skip connections let very deep networks train well. We use one pretrained on ImageNet. |
| Why fine-tune all layers? | The photos are different enough from ImageNet that adjusting every layer helps, and we use a small learning rate to stay safe. |
| What is a confidence score? | The softmax output: the model's probabilities across the classes, adding up to 1. Confident is not the same as correct. |
| Precision vs recall? | Precision: of the photos called "metal", how many really were. Recall: of all real metal photos, how many we found. |
| How do you know it is not overfitting? | Validation and test photos are never used for learning, and the test set is scored once. Training curves show validation levelling off. |
| Why only 93%? | Small clean dataset; plastic and metal look alike in some photos. We do not claim more. |
| Why use Gemini if you trained a model? | The model knows 6 materials and can be confidently wrong. Gemini finds items, double-checks, and covers everything else. Together: 84% vs 63%. |
| What is function calling? | The model asks for a named function to run with arguments; our server runs it and returns the result. |
| Could the agent make things up? | The numbers (bins, recovery, CO₂) come from our code and knowledge base, not from the model. The model writes the explanation. |
| How does the optimizer decide? | Simple rules: light dirt is recoverable by rinsing, heavy grease on paper or cardboard salvages half, other heavy contamination is lost. |
| How accurate is the CO₂ number? | It is an estimate from typical weights and rough factors, and the app says so. |
| What if the API is down or out of quota? | Each call times out in 20 s and the app moves to the next of 4 models. |
| Is the API key safe? | It is stored as a secret variable on the host and in a private file locally, never in the repository. |
| Where is it deployed? | Packaged with Docker, code on GitHub, deployed on Railway from the repo. |
| What would you improve next? | More and messier training photos, proper tests for textiles and hazardous items, a mobile app, and a permanent paid hosting plan. |

---

## 7. Do not claim, and do say

| Do not say | Say instead |
|---|---|
| "It is 100% accurate" or "it works for any waste" | "93% on our test photos, about 84% on outside photos with the agent." |
| "The CNN alone handles everything" | "The classifier is one tool; the agent checks it." |
| "The CO₂ figures are exact" | "They are estimates." |
| "We used Hugging Face" | "We deployed on Railway." |
| "Hazardous detection is solid" | "It was checked on a small number of photos." |

---

## 8. If you are asked to show code

| Question | Open |
|---|---|
| How is the model trained? | `src/train.py` (or the Colab notebook) |
| How does the agent work? | `src/agent.py`: the three tool functions and the system prompt |
| How does the optimizer work? | `src/optimizer.py` |
| How is the classifier used? | `src/classifier.py` |
| Where is the knowledge base? | `src/knowledge.py` |
| How does the live progress work? | `src/api.py`: the `/api/chat/stream` route |
