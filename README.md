<div align="center">

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/CogitoNTNU/DeepTactics-TrackMania/ci.yml)
![GitHub top language](https://img.shields.io/github/languages/top/CogitoNTNU/DeepTactics-TrackMania)
![GitHub language count](https://img.shields.io/github/languages/count/CogitoNTNU/DeepTactics-TrackMania)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Version](https://img.shields.io/badge/version-0.0.1-blue)](https://img.shields.io/badge/version-0.0.1-blue)

<img src="docs/images/trackmania.png" width="50%" alt="Cogito Project Logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>

<details>
<summary><b>📋 Table of contents </b></summary>

- [DeepTactics-TrackMania](#deeptactics-trackmania)
  - [🎯 Project Goals](#-project-goals)
  - [🧠 Project Description](#-project-description)
  - [🏗️ Architecture \& Tech Stack](#️-architecture--tech-stack)
  - [📚 Key Resources](#-key-resources)
  - [🚀 Getting Started](#-getting-started)
  - [Description](#description)
  - [🛠️ Prerequisites](#️-prerequisites)
  - [Getting started](#getting-started)
  - [Usage](#usage)
    - [📖 Generate Documentation Site](#-generate-documentation-site)
  - [Testing](#testing)
  - [👥 Team](#-team)
    - [Project Leads](#project-leads)
    - [License](#license)

</details>

# DeepTactics-TrackMania

🚗 **Deeptactics Trackmania** is a student-driven project exploring **Reinforcement Learning (RL)** in the racing game **Trackmania**.  
Our goal is to design, train, and visualize agents capable of completing tracks, improving over time, and eventually outperforming human players in our group.

---

## 🎯 Project Goals

- **Main Goal:**  
  Build an RL system that can successfully complete a Trackmania track.

- **Subgoals:**  
  - Achieve podium placement within the group.  
  - Beat all group members on at least one track.  
  - Visualize the agent inside the game.  
  - Ensure all members gain hands-on RL experience.  
  - Understand **exploration vs exploitation**.  
  - Enable everyone to start training runs on their own PC.  
  - Promote collaboration: *every line of code should be understood by the group*.  
  - Document progress with a short film showing the agent’s improvement.  

---

## 🧠 Project Description

We aim to train RL agents using a variety of methods (PPO, DQN variants, SAC, IQN, etc.) in **Trackmania**.  
The project emphasizes:  

- Experimenting with multiple RL approaches.  
- Building shared knowledge through research workshops.  
- Using visualization and metrics dashboards (e.g. WandB) to monitor progress.  
- Combining technical learning with social team-building.  

---

## 🏗️ Architecture & Tech Stack

- **Environment:** [Gymnasium](https://gymnasium.farama.org) + [TMRL](https://github.com/trackmania-rl/tmrl) / [TMInterface](https://donadigo.com/tminterface/)  
- **RL Algorithms:** PPO, SAC, DQN variants (including IQN)  
- **Experiment Tracking:** Weights & Biases (WandB)  
- **Tooling:** Git, Docker, GitHub Actions, pre-commit  
- **Visualization:** Custom render scripts for agent playback  

---

## 📚 Key Resources

- [PPO Paper (Schulman et al.)](https://arxiv.org/abs/1707.06347)  
- [SAC Paper (Haarnoja et al.)](https://arxiv.org/abs/1801.01290)  
- [IQN Paper](https://arxiv.org/abs/1806.06923)  
- [TMRL Framework](https://github.com/trackmania-rl/tmrl)  
- [Linesight RL (YouTube)](https://www.youtube.com/@linesight-rl)  
- [TMUnlimiter](https://unlimiter.net/)  

---

## 🚀 Getting Started

1. Clone repo & install dependencies (Docker setup provided).  
2. Configure environment (Gymnasium + Trackmania interface).  
3. Run baseline PPO agent training.  
4. Track results in WandB and visualize in-game.  

## Description

<!-- TODO: Provide a brief overview of what this project does and its key features. Please add pictures or videos of the application -->

## 🛠️ Prerequisites

<!-- TODO: In this section you put what is needed for the program to run.
For example: OS version, programs, libraries, etc.

-->

- **Git**: Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- **Python 3.12**: Required for the project. [Download Python](https://www.python.org/downloads/)
- **UV**: Used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** (optional): For DevContainer development. [Download Docker](https://www.docker.com/products/docker-desktop)

## Getting started

<!-- TODO: In this Section you describe how to install this project in its intended environment.(i.e. how to get it to run)
-->

1. **Clone the repository**:

   ```sh
   git clone https://github.com/CogitoNTNU/DeepTactics-TrackMania.git
   cd DeepTactics-TrackMania
   ```

1. **Install dependencies**:

   ```sh
   uv sync
   ```

<!--
1. **Configure environment variables**:
    This project uses environment variables for configuration. Copy the example environment file to create your own:
    ```sh
    cp .env.example .env
    ```
    Then edit the `.env` file to include your specific configuration settings.
-->

1. **Set up pre commit** (only for development):

   ```sh
   uv run pre-commit install
   ```

## Usage

To run the project, run the following command from the root directory of the project:

```bash

```

<!-- TODO: Instructions on how to run the project and use its features. -->

### 📖 Generate Documentation Site

To build and preview the documentation site locally:

```bash
uv run mkdocs build
uv run mkdocs serve
```

This will build the documentation and start a local server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) where you can browse the docs and API reference. Get the documentation according to the lastes commit on main by viewing the `gh-pages` branch on GitHub: [https://cogitontnu.github.io/DeepTactics-TrackMania/](https://cogitontnu.github.io/DeepTactics-TrackMania/).

## Testing

To run the test suite, run the following command from the root directory of the project:

```bash
uv run pytest --doctest-modules --cov=src --cov-report=html
```
## 👥 Team

This project would not have been possible without the hard work and dedication of all of the contributors.  
Thank you for the time and effort you have put into making this project a reality.

### Project Leads

<table align="center">
    <tr>
        <td align="center">
            <a href="https://github.com/ludvigovrevik">
              <img src="https://github.com/ludvigovrevik.png?size=100" width="100px;" alt="Ludvig Øvrevik"/><br />
              <sub><b>Ludvig Øvrevik</b></sub>
            </a><br />
            <sub>Project Lead</sub>
        </td>
        <td align="center">
            <a href="https://github.com/bragekvamme">
              <img src="https://github.com/brageHK.png?size=100" width="100px;" alt="Brage Kvamme"/><br />
              <sub><b>Brage Kvamme</b></sub>
            </a><br />
            <sub>Project Lead</sub>
        </td>
        <td align="center">
            <a href="https://github.com/edvard-ek">
              <img src="https://github.com/edvard-ek.png?size=100" width="100px;" alt="Edvard Klavenes"/><br />
              <sub><b>Edvard Klavenes</b></sub>
            </a><br />
            <sub>Project Member</sub>
        </td>
    </tr>
</table>


![Group picture](docs/img/team.png)

### License

---

Distributed under the MIT License. See `LICENSE` for more information.
