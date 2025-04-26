# KartalOl-HID-2025

This repository contains our preliminary work for the **6th International Competition on Human Identification at a Distance (HID 2025)**. Due to limited resources, we were unable to continue developing more advanced ideas or a better-performing model.

However, we are sharing a basic model related to HID, along with instructions on how to set up and run it. If you have access to better resources, you can build upon our work — we will also provide guidance to help you prepare for the competition evaluation.

Competition Website: [HID 2025 - CodaLab](https://codalab.lisn.upsaclay.fr/competitions/21845#learn_the_details)

---

## How to Run the Model

1. **Needed package to install**
   ```torch
   torchvision
   numpy
   PIL
   sklearn
   ```

2. **Prepare the Dataset**  
   You will need to gather the necessary datasets, some of which are publicly available.

3. **Set the Dataset Path**  
   Update the dataset path in your local environment accordingly.

4. **Training**  
   Run the following command to train the baseline model:

   ```bash
   python script/baseline_train.py
   ```

4. **Inference**  
   After training, run the inference script:

   ```bash
   python script/baseline_inferense.py
   ```

---

Feel free to clone the repo, enhance the model, and use it as a starting point for the competition.  
Good luck!

---

@misc{KartalOl-HID-2025,
  author       = {Jalil Nourmohammadi Khiarak and Taher Akbari Saeed and Ali Kianfar and Mahsa Nasehi and Mobina Pashazadeh Panahi},
  title        = {KartalOl-HID-2025: A Simple Model for the 6th International Competition on Human Identification at a Distance (HID 2025)},
  year         = {2025},
  howpublished = {\url{https://github.com/Jalilnkh/KartalOl-HID-2025}},
  note         = {Preliminary work shared for the HID 2025 competition.}
}

