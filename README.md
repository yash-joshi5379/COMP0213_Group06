# GraspPlanner

This repository contains all the Python code and supporting files used for our grasp planning project. The project is split into separate folders to provide simple navigation through all of our files and classes.

> ✅ **Plug-and-play ready** — just download the full repository, install the requirements and you're ready!
---

## 🧭 Project Structure
| Folder                 | Purpose                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **classification**     | Contains all model-training and classification logic, including classifiers, model selection, and evaluation code. |
| **confusion_matrices** | Stores generated confusion matrix images for evaluating model performance.                                         |
| **datasets**           | Holds generated or collected grasp-trial datasets used for training and testing models.                            |
| **figures**            | Contains plots, charts, or other figures produced during analysis or evaluation.                                   |
| **grippers**           | Includes gripper configuration files or classes representing different robotic gripper types.                      |
| **interface**          | Houses the PyQt6 GUI code, including windows, dialogs, and threading utilities for the app.                        |
| **models**             | Stores saved machine-learning models (e.g., `.pkl` files, neural network weights) generated during classification. |
| **objects**            | Contains object definition files, meshes, or metadata used in simulation grasp trials.                             |
| **simulation**         | Implements the simulation environment, physics setup, and grasp trial execution logic.                             |
| **urdf_files**         | URDF robot/gripper description files used by the simulator for accurate physical modelling.                        |

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
Clone this repository or download it to your local computer.

```bash
git clone https://github.com/yash-joshi5379/GraspPlanner
```

### 2. Install all Required Libraries
Run the following instruction in the command line:

```bash
pip install -r requirements.txt
```
### 3. Run the file named 'main.py'
You should see a GUI appear, feel free to explore from there!

![alt text](https://github.com/yash-joshi5379/GraspPlanner/blob/main/GUI_image%20(1).png "GUI_image")
