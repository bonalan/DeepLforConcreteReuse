<img width="1087" height="283" alt="Screenshot 2025-11-17 at 13 59 01" src="https://github.com/user-attachments/assets/bf516909-e604-47c1-821a-176f58aba9d2" />

## Deep Neural Network-Based Design Exploration with Concrete Cutting Waste

This repository contains supplementary material for the publication **Deep Neural Network-Based Design Exploration with Concrete Cutting Waste**, by Beril Önalan, Eleftherios Triantafyllidis, Ioanna Mitropoulou, and Catherine De Wolf, published in *Technology | Architecture + Design* (2025).

Our research introduces a computational design approach to automate the reuse of concrete cutting waste in architectural elements during the early design phase. The methodology employs a deep learning-based workflow to facilitate performance-based design from a constrained stock of concrete waste, enabling scalability and efficiency in circular design practices.

**Paper**: [https://doi.org/10.1080/24751448.2025.2534788](https://doi.org/10.1080/24751448.2025.2534788)

---

## Features

- **Synthetic Dataset**: Excel datasheet derived from the parametric model (10,080 design instances)
- **Deep Learning Model**: Surrogate model for rapid performance prediction (R² = 0.90, 127-190x speedup)
- **Design Exploration Tools**: API implementation for real-time design space exploration

---

## Repository Contents

- `/data`: Synthetic dataset for training and testing the deep neural network
- `/model`: Pre-trained model, scalers, and training script for the surrogate model
- `/API_call`: Function for the API call

---

## Citation

If you use this repository, please cite:
```bibtex
@article{onalan2025deep,
  title={Deep Neural Network-Based Design Exploration with Concrete Cutting Waste},
  author={\"Onalan, Beril and Triantafyllidis, Eleftherios and Mitropoulou, Ioanna and De Wolf, Catherine},
  journal={Technology | Architecture + Design},
  year={2025},
  doi={10.1080/24751448.2025.2534788},
  url={https://doi.org/10.1080/24751448.2025.2534788}
}
```

---

## Contact

For questions, please contact: [boenalan@ethz.ch](mailto:boenalan@ethz.ch)

---

## License

This work is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). You are free to share and adapt the material with appropriate attribution and under the same license.
