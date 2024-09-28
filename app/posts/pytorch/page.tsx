"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { ChevronUp, Clock, Share2, Check } from "lucide-react";

export default function CreatingAIWithPyTorch() {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const totalHeight =
        document.documentElement.scrollHeight -
        document.documentElement.clientHeight;
      const progress = (window.scrollY / totalHeight) * 100;
      setScrollProgress(progress);
      setShowScrollTop(window.scrollY > 300);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error("Failed to copy: ", err);
    }
  };

  return (
    <article className="relative space-y-6 text-xs">
      <div className="fixed top-0 left-0 w-full h-1 bg-gray-200">
        <div
          className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
          style={{ width: `${scrollProgress}%` }}
        ></div>
      </div>

      <Link
        href="/"
        className="text-blue-400 hover:underline inline-block mb-4"
      >
        {"<"} Back to home
      </Link>

      <header className="space-y-2">
        <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500">
          Creating an AI with PyTorch: A Comprehensive Beginner's Guide
        </h1>
        <p className="text-gray-400">
          Embark on an in-depth journey into artificial intelligence, mastering
          PyTorch to build and train your own neural network
        </p>
        <div className="flex items-center space-x-4 text-gray-500">
          <span className="flex items-center">
            <Clock size={12} className="mr-1" /> 15 min read
          </span>
          <button
            onClick={handleShare}
            className="flex items-center hover:text-gray-300 transition-colors relative"
          >
            {copied ? (
              <Check size={12} className="mr-1" />
            ) : (
              <Share2 size={12} className="mr-1" />
            )}
            {copied ? "Copied!" : "Share"}
            {copied && (
              <span className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white px-2 py-1 rounded text-xs">
                Link copied!
              </span>
            )}
          </button>
        </div>
      </header>

      <nav className="border border-gray-800 rounded p-4">
        <h2 className="font-semibold mb-2">Table of Contents</h2>
        <ul className="space-y-1">
          <li>
            <a
              href="#introduction"
              className="hover:text-blue-400 transition-colors"
            >
              Introduction
            </a>
          </li>
          <li>
            <a
              href="#prerequisites"
              className="hover:text-blue-400 transition-colors"
            >
              Prerequisites and Setup
            </a>
          </li>
          <li>
            <a
              href="#understanding-nn"
              className="hover:text-blue-400 transition-colors"
            >
              Understanding Neural Networks
            </a>
          </li>
          <li>
            <a href="#step1" className="hover:text-blue-400 transition-colors">
              Step 1: Importing Libraries and Preparing Data
            </a>
          </li>
          <li>
            <a href="#step2" className="hover:text-blue-400 transition-colors">
              Step 2: Defining the Neural Network Architecture
            </a>
          </li>
          <li>
            <a href="#step3" className="hover:text-blue-400 transition-colors">
              Step 3: Defining Loss Function and Optimizer
            </a>
          </li>
          <li>
            <a href="#step4" className="hover:text-blue-400 transition-colors">
              Step 4: Training the Network
            </a>
          </li>
          <li>
            <a href="#step5" className="hover:text-blue-400 transition-colors">
              Step 5: Evaluating the Model
            </a>
          </li>
          <li>
            <a href="#step6" className="hover:text-blue-400 transition-colors">
              Step 6: Visualizing Results and Model Interpretation
            </a>
          </li>
          <li>
            <a
              href="#advanced-topics"
              className="hover:text-blue-400 transition-colors"
            >
              Advanced Topics and Further Learning
            </a>
          </li>
          <li>
            <a
              href="#conclusion"
              className="hover:text-blue-400 transition-colors"
            >
              Conclusion
            </a>
          </li>
        </ul>
      </nav>

      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 800 400"
        className="w-full h-auto"
      >
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="0"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#ffffff" />
          </marker>
        </defs>

        <rect width="100%" height="100%" fill="#000000" />

        <text
          x="400"
          y="30"
          fontSize="24"
          fill="#ffffff"
          textAnchor="middle"
          fontWeight="bold"
        >
          Neural Network Architecture
        </text>

        <g transform="translate(50, 100)">
          <rect
            width="100"
            height="200"
            fill="#1a1a1a"
            stroke="#ffffff"
            strokeWidth="2"
          />
          <text x="50" y="-10" fontSize="16" fill="#ffffff" textAnchor="middle">
            Input Layer
          </text>
          <circle cx="50" cy="40" r="10" fill="#4a4a4a" />
          <circle cx="50" cy="100" r="10" fill="#4a4a4a" />
          <circle cx="50" cy="160" r="10" fill="#4a4a4a" />
          <text x="50" y="220" fontSize="14" fill="#ffffff" textAnchor="middle">
            (28 x 28 pixels)
          </text>
        </g>

        <g transform="translate(250, 100)">
          <rect
            width="100"
            height="200"
            fill="#1a1a1a"
            stroke="#ffffff"
            strokeWidth="2"
          />
          <text x="50" y="-10" fontSize="16" fill="#ffffff" textAnchor="middle">
            Hidden Layer 1
          </text>
          <circle cx="50" cy="40" r="10" fill="#4a4a4a" />
          <circle cx="50" cy="100" r="10" fill="#4a4a4a" />
          <circle cx="50" cy="160" r="10" fill="#4a4a4a" />
          <text x="50" y="220" fontSize="14" fill="#ffffff" textAnchor="middle">
            (128 neurons)
          </text>
        </g>

        <g transform="translate(450, 100)">
          <rect
            width="100"
            height="200"
            fill="#1a1a1a"
            stroke="#ffffff"
            strokeWidth="2"
          />
          <text x="50" y="-10" fontSize="16" fill="#ffffff" textAnchor="middle">
            Hidden Layer 2
          </text>
          <circle cx="50" cy="40" r="10" fill="#4a4a4a" />
          <circle cx="50" cy="100" r="10" fill="#4a4a4a" />
          <circle cx="50" cy="160" r="10" fill="#4a4a4a" />
          <text x="50" y="220" fontSize="14" fill="#ffffff" textAnchor="middle">
            (64 neurons)
          </text>
        </g>

        <g transform="translate(650, 100)">
          <rect
            width="100"
            height="200"
            fill="#1a1a1a"
            stroke="#ffffff"
            strokeWidth="2"
          />
          <text x="50" y="-10" fontSize="16" fill="#ffffff" textAnchor="middle">
            Output Layer
          </text>
          <circle cx="50" cy="40" r="10" fill="#4a4a4a" />
          <circle cx="50" cy="100" r="10" fill="#4a4a4a" />
          <circle cx="50" cy="160" r="10" fill="#4a4a4a" />
          <text x="50" y="220" fontSize="14" fill="#ffffff" textAnchor="middle">
            (10 neurons)
          </text>
        </g>

        <g stroke="#ffffff" strokeWidth="1" opacity="0.5">
          <line
            x1="150"
            y1="140"
            x2="250"
            y2="140"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="150"
            y1="140"
            x2="250"
            y2="200"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="150"
            y1="200"
            x2="250"
            y2="140"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="150"
            y1="200"
            x2="250"
            y2="200"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="150"
            y1="260"
            x2="250"
            y2="200"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="150"
            y1="260"
            x2="250"
            y2="260"
            markerEnd="url(#arrowhead)"
          />

          <line
            x1="350"
            y1="140"
            x2="450"
            y2="140"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="350"
            y1="140"
            x2="450"
            y2="200"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="350"
            y1="200"
            x2="450"
            y2="140"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="350"
            y1="200"
            x2="450"
            y2="200"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="350"
            y1="260"
            x2="450"
            y2="200"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="350"
            y1="260"
            x2="450"
            y2="260"
            markerEnd="url(#arrowhead)"
          />

          <line
            x1="550"
            y1="140"
            x2="650"
            y2="140"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="550"
            y1="140"
            x2="650"
            y2="200"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="550"
            y1="200"
            x2="650"
            y2="140"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="550"
            y1="200"
            x2="650"
            y2="200"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="550"
            y1="260"
            x2="650"
            y2="200"
            markerEnd="url(#arrowhead)"
          />
          <line
            x1="550"
            y1="260"
            x2="650"
            y2="260"
            markerEnd="url(#arrowhead)"
          />
        </g>
      </svg>

      <section id="introduction" className="space-y-4">
        <h2 className="text-sm font-semibold">Introduction</h2>
        <p>
          Artificial Intelligence (AI) has revolutionized numerous fields, from
          healthcare to finance, entertainment to transportation. At the heart
          of many AI systems are neural networks, powerful models inspired by
          the human brain. In this comprehensive guide, we'll dive deep into the
          world of AI, using PyTorch - a leading deep learning framework - to
          create a sophisticated neural network capable of recognizing
          handwritten digits.
        </p>
        <p>
          This tutorial is designed for beginners, but it doesn't shy away from
          the complexities of AI. We'll cover everything from the basics of
          neural networks to advanced techniques in data preprocessing, model
          architecture, training optimization, and result visualization. By the
          end, you'll have built a robust AI model and gained the knowledge to
          tackle more complex AI projects.
        </p>
      </section>

      <section id="prerequisites" className="space-y-4">
        <h2 className="text-sm font-semibold">Prerequisites and Setup</h2>
        <p>
          Before we begin our AI journey, let's ensure we have the right tools
          and environment:
        </p>
        <div className="space-y-2">
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold">1. Python Environment</h3>
            <p>
              We'll be using Python 3.8 or later. If you haven't installed
              Python, download it from the{" "}
              <a
                href="https://www.python.org/downloads/"
                className="text-blue-400 hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                official Python website
              </a>
              .
            </p>
          </div>

          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold">2. Required Libraries</h3>
            <p>We'll need the following libraries:</p>
            <ul className="list-disc list-inside">
              <li>PyTorch: Our main deep learning framework</li>
              <li>
                torchvision: For easy access to datasets and common model
                architectures
              </li>
              <li>NumPy: For numerical computing</li>
              <li>Matplotlib: For visualizing our results</li>
              <li>tqdm: For progress bars during training</li>
            </ul>
            <p>Install these packages using pip:</p>
            <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
              <code>pip install torch torchvision numpy matplotlib tqdm</code>
            </pre>
          </div>

          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold">
              3. GPU Support (Optional but Recommended)
            </h3>
            <p>
              If you have a CUDA-capable NVIDIA GPU, you can significantly speed
              up training. Follow the{" "}
              <a
                href="https://pytorch.org/get-started/locally/"
                className="text-blue-400 hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                PyTorch installation guide
              </a>{" "}
              to install the GPU-enabled version.
            </p>
          </div>

          <div
            className="border border-gray-800 rounde
d p-3 hover:bg-gray-900 transition-colors"
          >
            <h3 className="font-semibold">4. Development Environment</h3>
            <p>
              While you can use any text editor, we recommend using an
              Integrated Development Environment (IDE) like{" "}
              <a
                href="https://code.visualstudio.com/"
                className="text-blue-400 hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                Visual Studio Code
              </a>{" "}
              with the Python extension for a better coding experience.
            </p>
          </div>
        </div>
      </section>

      <section id="understanding-nn" className="space-y-4">
        <h2 className="text-sm font-semibold">Understanding Neural Networks</h2>
        <p>
          Before we dive into coding, let's briefly explore what neural networks
          are and how they work.
        </p>
        <div className="space-y-2">
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold">What is a Neural Network?</h3>
            <p>
              A neural network is a computational model inspired by the human
              brain. It consists of interconnected nodes (neurons) organized in
              layers. The basic structure includes:
            </p>
            <ul className="list-disc list-inside">
              <li>Input Layer: Receives the initial data</li>
              <li>Hidden Layers: Process the data</li>
              <li>Output Layer: Produces the final result</li>
            </ul>
            <p>
              As shown in the diagram above, our neural network for handwritten
              digit recognition will have an input layer of 784 neurons (28x28
              pixels), two hidden layers with 128 and 64 neurons respectively,
              and an output layer with 10 neurons (one for each digit from 0 to
              9).
            </p>
          </div>
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold">How Neural Networks Learn</h3>
            <p>
              Neural networks learn through a process called backpropagation.
              Here's a simplified explanation:
            </p>
            <ol className="list-decimal list-inside">
              <li>The network makes a prediction based on input data</li>
              <li>
                The prediction is compared to the actual result, calculating the
                error
              </li>
              <li>The error is propagated backwards through the network</li>
              <li>The network's weights are adjusted to minimize this error</li>
              <li>
                This process is repeated many times with different data points
              </li>
            </ol>
          </div>
        </div>
      </section>

      <section id="step1" className="space-y-4">
        <h2 className="text-sm font-semibold">
          Step 1: Importing Libraries and Preparing Data
        </h2>
        <p>
          Let's start by importing the necessary libraries and loading our
          dataset. We'll use the MNIST dataset, a large collection of
          handwritten digits that is commonly used for training various image
          processing systems.
        </p>
        <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
          <code>
            {`import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Visualize some images
def imshow(img):
  img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

# Get random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{labels[j]:5d}' for j in range(4)))`}
          </code>
        </pre>
        <p>In this code, we're doing several important things:</p>
        <ol className="list-decimal list-inside">
          <li>Importing necessary libraries</li>
          <li>Setting up CUDA for GPU acceleration if available</li>
          <li>Defining data transformations to normalize our images</li>
          <li>Loading the MNIST dataset from torchvision</li>
          <li>
            Creating DataLoader objects for efficient batching and shuffling
          </li>
          <li>Defining a function to visualize our images</li>
        </ol>
        <p>
          The MNIST dataset is automatically downloaded if it's not already
          present in the ./data directory. This dataset contains 60,000 training
          images and 10,000 test images, each 28x28 pixels in size.
        </p>
      </section>

      <section id="step2" className="space-y-4">
        <h2 className="text-sm font-semibold">
          Step 2: Defining the Neural Network Architecture
        </h2>
        <p>
          Now, let's define our neural network architecture. We'll create a more
          sophisticated network than a simple feedforward one, incorporating
          convolutional layers for better feature extraction.
        </p>
        <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
          <code>
            {`class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
      x = self.conv1(x)
      x = nn.functional.relu(x)
      x = self.conv2(x)
      x = nn.functional.relu(x)
      x = nn.functional.max_pool2d(x, 2)
      x = self.dropout1(x)
      x = torch.flatten(x, 1)
      x = self.fc1(x)
      x = nn.functional.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)
      output = nn.functional.log_softmax(x, dim=1)
      return output

model = Net().to(device)
print(model)`}
          </code>
        </pre>
        <p>Let's break down this architecture:</p>
        <ul className="list-disc list-inside">
          <li>
            Two convolutional layers (conv1 and conv2) for feature extraction
          </li>
          <li>Max pooling layer to reduce spatial dimensions</li>
          <li>Dropout layers to prevent overfitting</li>
          <li>Two fully connected layers (fc1 and fc2) for classification</li>
          <li>ReLU activation functions for non-linearity</li>
          <li>Log softmax for the output layer</li>
        </ul>
        <p>
          This network is more complex than a simple fully connected network and
          is better suited for image recognition tasks.
        </p>
      </section>

      <section id="step3" className="space-y-4">
        <h2 className="text-sm font-semibold">
          Step 3: Defining Loss Function and Optimizer
        </h2>
        <p>
          For our loss function, we'll use Negative Log Likelihood Loss, which
          works well with our log softmax output. For optimization, we'll use
          Adam, an adaptive learning rate optimization algorithm.
        </p>
        <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
          <code>
            {`criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)`}
          </code>
        </pre>
        <p>
          We've also added a learning rate scheduler, which will decrease the
          learning rate by a factor of 0.7 every epoch. This can help in
          fine-tuning the model as training progresses.
        </p>
      </section>

      <section id="step4" className="space-y-4">
        <h2 className="text-sm font-semibold">Step 4: Training the Network</h2>
        <p>
          Now, let's train our network. We'll iterate over the dataset multiple
          times (epochs) and update our network's parameters. We'll also
          implement early stopping to prevent overfitting.
        </p>
        <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
          <code>
            {`def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  for batch_idx, (data, target) in enumerate(pbar):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      pbar.set_description(desc= f'Loss: {loss.item():.4f}')

def test(model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          test_loss += criterion(output, target).item()
          pred = output.argmax(dim=1, keepdim=True)
          correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  print(f'\
Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\
')
  return test_loss

# Training loop
n_epochs = 10
best_loss = float('inf')
patience = 3
trigger_times = 0

for epoch in range(1, n_epochs + 1):
  train(model, device, trainloader, optimizer, epoch)
  test_loss = test(model, device, testloader)
  scheduler.step()
  
  if test_loss < best_loss:
      trigger_times = 0
      best_loss = test_loss
  else: 
      trigger_times += 1
      if trigger_times >= patience:
          print('Early stopping!')
          break

print('Finished Training')`}
          </code>
        </pre>
        <p>This training loop includes several advanced features:</p>
        <ul className="list-disc list-inside">
          <li>
            Progress bar using tqdm for better visualization of training
            progress
          </li>
          <li>Separate train and test functions for cleaner code</li>
          <li>Early stopping to prevent overfitting</li>
          <li>Learning rate scheduling</li>
        </ul>
      </section>

      <section id="step5" className="space-y-4">
        <h2 className="text-sm font-semibold">Step 5: Evaluating the Model</h2>
        <p>
          After training, let's evaluate our model's performance more
          thoroughly.
        </p>
        <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
          <code>
            {`def evaluate(model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          test_loss += criterion(output, target).item()
          pred = output.argmax(dim=1, keepdim=True)
          correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  accuracy = 100. * correct / len(test_loader.dataset)
  return test_loss, accuracy

test_loss, accuracy = evaluate(model, device, testloader)
print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = []
y_true = []

model.eval()
with torch.no_grad():
  for data, target in testloader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      pred = output.argmax(dim=1, keepdim=True)
      y_pred.extend(pred.view(-1).cpu().numpy())
      y_true.extend(target.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()`}
          </code>
        </pre>
        <p>This evaluation includes:</p>
        <ul className="list-disc list-inside">
          <li>A more detailed evaluation function</li>
          <li>
            Calculation of a confusion matrix to see where our model is making
            mistakes
          </li>
        </ul>
      </section>

      <section id="step6" className="space-y-4">
        <h2 className="text-sm font-semibold">
          Step 6: Visualizing Results and Model Interpretation
        </h2>
        <p>
          Finally, let's visualize some of our model's predictions and try to
          interpret its decision-making process.
        </p>
        <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
          <code>
            {`def visualize_predictions(model, device, test_loader, num_images=10):
  model.eval()
  images_so_far = 0
  fig = plt.figure(figsize=(15, 15))
  
  with torch.no_grad():
      for i, (data, target) in enumerate(test_loader):
          data, target = data.to(device), target.to(device)
          outputs = model(data)
          _, preds = torch.max(outputs, 1)
          
          for j in range(data.size()[0]):
              images_so_far += 1
              ax = plt.subplot(5, 5, images_so_far)
              ax.axis('off')
              ax.set_title(f'Predicted: {preds[j]}, Actual: {target[j]}')
              imshow(data.cpu().data[j])
              
              if images_so_far == num_images:
                  return
  plt.tight_layout()
  plt.show()

visualize_predictions(model, device, testloader)

# Visualizing feature maps
def visualize_feature_maps(model, device, test_loader):
  model.eval()
  with torch.no_grad():
      data, _ = next(iter(test_loader))
      data = data.to(device)
      
      # Get feature maps from first conv layer
      feature_maps = model.conv1(data)
      
      fig = plt.figure(figsize=(20, 10))
      for i in range(32):
          ax = fig.add_subplot(4, 8, i+1)
          ax.imshow(feature_maps[0, i].cpu().numpy(), cmap='gray')
          ax.axis('off')
      plt.tight_layout()
      plt.show()

visualize_feature_maps(model, device, testloader)`}
          </code>
        </pre>
        <p>These visualizations help us understand:</p>
        <ul className="list-disc list-inside">
          <li>How well our model is performing on individual examples</li>
          <li>What kind of features our convolutional layers are detecting</li>
        </ul>
      </section>

      <section id="advanced-topics" className="space-y-4">
        <h2 className="text-sm font-semibold">
          Advanced Topics and Further Learning
        </h2>
        <p>
          Congratulations! You've built a sophisticated AI model using PyTorch.
          But this is just the beginning. Here are some advanced topics you can
          explore next:
        </p>
        <ul className="list-disc list-inside space-y-2">
          <li>
            <strong>Transfer Learning:</strong> Use pre-trained models to solve
            complex tasks with less data. Check out{" "}
            <a
              href="https://huggingface.co/models"
              className="text-blue-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              Hugging Face's model hub
            </a>{" "}
            for a wide range of pre-trained models.
          </li>
          <li>
            <strong>Generative Adversarial Networks (GANs):</strong> Learn how
            to generate new, synthetic data that looks real.
          </li>
          <li>
            <strong>Reinforcement Learning:</strong> Train agents to make
            decisions in complex environments.
          </li>
          <li>
            <strong>Natural Language Processing:</strong> Apply deep learning to
            text data. The{" "}
            <a
              href="https://huggingface.co/datasets"
              className="text-blue-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              Hugging Face datasets
            </a>{" "}
            library is a great resource for NLP datasets.
          </li>
          <li>
            <strong>Deployment:</strong> Learn how to deploy your models in
            production environments using tools like{" "}
            <a
              href="https://www.tensorflow.org/tfx"
              className="text-blue-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              TensorFlow Extended (TFX)
            </a>{" "}
            or{" "}
            <a
              href="https://mlflow.org/"
              className="text-blue-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              MLflow
            </a>
            .
          </li>
        </ul>
      </section>

      <section id="conclusion" className="space-y-4">
        <h2 className="text-sm font-semibold">Conclusion</h2>
        <p>
          In this comprehensive guide, we've journeyed from the basics of neural
          networks to building and training a sophisticated AI model using
          PyTorch. We've covered data preparation, model architecture, training
          optimization, and result visualization. This knowledge forms a solid
          foundation for tackling more complex AI projects and diving deeper
          into the fascinating world of artificial intelligence.
        </p>
        <p>
          Remember, the field of AI is vast and rapidly evolving. Stay curious,
          keep experimenting, and never stop learning. The AI revolution is just
          beginning, and you're now equipped to be a part of it!
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-sm font-semibold">Additional Resources</h2>
        <ul className="list-disc list-inside space-y-2">
          <li>
            <a
              href="https://pytorch.org/tutorials/"
              className="text-blue-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              Official PyTorch Tutorials
            </a>
          </li>
          <li>
            <a
              href="https://www.deeplearning.ai/"
              className="text-blue-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              DeepLearning.AI Courses
            </a>
          </li>
          <li>
            <a
              href="https://www.fast.ai/"
              className="text-blue-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              Fast.ai Practical Deep Learning Course
            </a>
          </li>
          <li>
            <a
              href="https://paperswithcode.com/"
              className="text-blue-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              Papers With Code
            </a>{" "}
            for staying up-to-date with the latest AI research
          </li>
          <li>
            <a
              href="https://github.com/pytorch/examples"
              className="text-blue-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              PyTorch Examples Repository
            </a>{" "}
            for more complex PyTorch implementations
          </li>
        </ul>
      </section>

      {showScrollTop && (
        <button
          onClick={scrollToTop}
          className="fixed bottom-4 right-4 bg-gray-800 p-2 rounded-full hover:bg-gray-700 transition-colors"
          aria-label="Scroll to top"
        >
          <ChevronUp size={20} />
        </button>
      )}
    </article>
  );
}
