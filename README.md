# KartalOl-HID-2025

This repository contains our preliminary work for the **6th International Competition on Human Identification at a Distance (HID 2025)**. Due to limited resources, we were unable to continue developing more advanced ideas or a better-performing model.

However, we are sharing a basic model related to HID, along with instructions on how to set up and run it. If you have access to better resources, you can build upon our work — we will also provide guidance to help you prepare for the competition evaluation.

Competition Website: [HID 2025 - CodaLab](https://codalab.lisn.upsaclay.fr/competitions/21845#learn_the_details)

---

## How to Run the Model

1. **Prepare the Dataset**  
   You will need to gather the necessary datasets, some of which are publicly available.

2. **Set the Dataset Path**  
   Update the dataset path in your local environment accordingly.

3. **Training**  
   Run the following command to train the baseline model:

   ```bash
   python baseline_train.py
   ```

4. **Inference**  
   After training, run the inference script:

   ```bash
   python baseline_inferense.py
   ```

---

Feel free to clone the repo, enhance the model, and use it as a starting point for the competition.  
Good luck!
