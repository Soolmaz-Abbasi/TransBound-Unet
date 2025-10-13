# TransBound-Unet
Improved A-Line and B-Line Detection in Lung Ultrasound Using Deep Learning with Boundary-Aware Dice Loss
### Abstract
Lung ultrasound (LUS) is a non-invasive bedside imaging technique for diagnosing pulmonary conditions, especially in critical care settings. A-lines and B-lines are important features in LUS images that help to assess lung health and identify changes in lung tissue. However, accurately detecting and segmenting these lines remains challenging, due to their subtle blurred boundaries. To address this, we propose TransBound-UNet, a novel segmentation model that integrates a transformer-based encoder with boundary-aware Dice loss to enhance medical image segmentation. This loss function incorporates boundary-specific penalties into a hybrid Dice-BCE formulation, allowing for more accurate segmentation of critical structures. 

### Installation

Recommended: create a Python virtual environment.

```bash
# create venv (optional)
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows PowerShell

# install dependencies
pip install -r requirements.txt


### Data

The dataset used in the paper is available upon request from the corresponding author.
Place images and masks in the following structure:

data/
├── images/
└── masks/

###Results

![bioengineering-12-00311-g004](https://github.com/user-attachments/assets/f9aa8ede-4b7d-4552-a502-5852f1fd1115)

