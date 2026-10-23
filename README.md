# Smart Crop Recommendation System

An educational Streamlit project for exploring how soil-feature patterns in a
public dataset relate to crop labels. It combines a calibrated, soil-only
classifier with live location weather displayed as context.

> **Educational prototype, not farming advice.** Predictions reflect one
> dataset's labels and coverage. They are not definitive crop suitability
> assessments, yield forecasts, or a replacement for soil testing and local
> agricultural expertise.

## Architecture

```text
app.py                         Streamlit interface
crop_app/
  crop_reference.py            crop summaries and credited Wikimedia images
  geography.py                 GeoNames India state/district/city hierarchy
  inference.py                 ranked predictions and local feature ablation
  modeling.py                  data loading, splitting, model comparison, calibration
  soil.py                      SoilGrids topsoil point estimates
  validation.py                input and model-range validation
  weather.py                   Open-Meteo current conditions
scripts/train_model.py         reproducible command-line training
notebooks/crop_model_workflow.ipynb
data/raw/                      manually downloaded dataset (not tracked)
artifacts/                     generated model and metadata (not tracked)
tests/                         focused unit tests
```

The app's light agriculture palette is configured in `.streamlit/config.toml`
so widget text, placeholders, selected states, and messages share a consistent
high-contrast theme rather than relying on CSS foreground/background overrides.

## Data, weather, and limitations

### Training dataset

- **Dataset:** [Crop Recommendation Dataset by Atharva Ingle on Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).
- **Declared license:** Apache 2.0, as shown in the [Kaggle dataset metadata](https://www.kaggle.com/api/v1/datasets/view/atharvaingle/crop-recommendation-dataset). Review the dataset page and license before reuse or redistribution.
- **Publisher's description:** the data were assembled by augmenting rainfall, climate, and fertilizer datasets available for India. It describes N, P, and K as soil-content ratios, temperature in °C, relative humidity in %, soil pH, and rainfall in mm.
- **Limitations:** the publisher does not provide complete source-level lineage or the time interval represented by rainfall, temperature, and humidity. In particular, rainfall's “mm” value has no documented accumulation period. N/P/K units are also unspecified. The dataset should not be treated as a validated agricultural recommendation standard.

The model uses **only N, P, K, and pH**. Weather features are intentionally excluded because their time scales and provenance cannot be matched safely to current weather. This prevents current conditions from being silently substituted for an undocumented seasonal or other climate summary.

Repeated rows with the same soil feature values and crop label are removed before splitting. Stratified, group-aware splits keep identical N/P/K/pH vectors together across training, validation, test, and calibration folds to reduce overlap leakage.

### Live weather

The app uses the [Open-Meteo forecast API's current conditions](https://open-meteo.com/en/docs) to show temperature (°C), relative humidity (%), and the provider's current-condition precipitation value (mm) for a selected location. Open-Meteo combines numerical weather-model output; these values are not guaranteed to be direct local-station observations. They are **context only and do not affect the model ranking**. Current precipitation is not interpreted as seasonal rainfall.

The cascading location menus use the daily [GeoNames India export](https://download.geonames.org/export/dump/IN.zip) and its [administrative division codes](https://download.geonames.org/export/dump/admin1CodesASCII.txt) and [district codes](https://download.geonames.org/export/dump/admin2Codes.txt). Country is India, followed by state/union territory, district, and populated city/town. The directory includes populated places with a reported population of at least 1,000 plus administrative seats; smaller or unrecorded places may be absent. GeoNames data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and must be attributed to [GeoNames](https://www.geonames.org/). GeoNames supplies the data “as is”; completeness and accuracy are not guaranteed. The location catalog is cached for 24 hours.

The selected populated place's GeoNames coordinates are used to query Open-Meteo. The [free Open-Meteo API](https://open-meteo.com/en/terms) is intended for non-commercial use, has published rate limits, and does not require an API key. Weather data are offered under [CC BY 4.0](https://open-meteo.com/en/licence); the app displays the required Open-Meteo attribution. If deploying commercially or beyond the free terms, review the provider's current terms and choose an appropriately licensed service. Weather responses are cached for ten minutes.

### Soil inputs and model interpretation

For the selected coordinate, the app queries [ISRIC SoilGrids 2.0](https://soilgrids.org/) via its [point-properties API](https://rest.isric.org/soilgrids/v2.0/docs). If present, its modelled **total nitrogen** and **pH in water** topsoil properties at 0–5 cm are displayed as point estimates. The API provides no observation date or temporal averaging period for a point estimate, so the app does not invent one. Results are cached for 24 hours. SoilGrids is a gridded model, not a field soil test or a district-wide average; its estimates may be absent at some coordinates.

SoilGrids does not provide phosphorus or potassium through this point API. Its total nitrogen value is reported in g/kg, while the crop dataset calls N a ratio but documents neither its units nor a compatible scale. Therefore the estimate is contextual only and is **never entered into the crop model**. An available pH estimate is prefilled as an editable starting point; users should confirm or replace it with a soil-test result. Users must provide N/P/K using the dataset's numeric scale, preferably from a soil test. If estimates are missing or the service is unavailable, no values are fabricated and the app asks for soil-test inputs. Inputs must fall inside the model's training ranges; out-of-range inputs are rejected instead of extrapolated.

The app shows the top three crops with their **sigmoid-calibrated model class probabilities**. These scores are model output among the dataset labels, not probabilities of crop success or measures of agronomic suitability. On the reproducible split in this project, top-label expected calibration error was 0.114 using 10 fixed-width bins, so treat the displayed scores cautiously. The notebook calibrates the selected model using cross-validation confined to the training partition and reports calibration diagnostics. Local factor explanations replace one submitted soil feature at a time with its training median and compare the selected crop's model probability. This is a model-sensitivity aid, not causal evidence.

Recommendation cards include a short crop overview from that crop's English Wikipedia page and a matching article image or explicitly mapped Commons crop image, with creator/license credit from Wikimedia Commons. See [Wikimedia Commons reuse guidance](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia) for license terms. A small crop profile shows the per-label median values recorded in the original training dataset (including its temperature and rainfall columns where available). These are dataset examples, **not optimal growing ranges, fertiliser recommendations, or measured local conditions**. The dataset publisher does not document N/P/K units or rainfall's accumulation period. The app links to the source crop article and gives a source credit for the training dataset profile. Crop article and image references are cached for 24 hours. If Wikimedia is unavailable or an image has no clear credit/license metadata, the app does not substitute an unrelated image.

## Setup

Use Python 3.11 or 3.12 and Streamlit 1.66 or newer (the dependency set is validated with Python 3.12). In PowerShell from the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download `Crop_recommendation.csv` from the Kaggle dataset page and place it at:

```text
data/raw/Crop_recommendation.csv
```

The dataset and trained artifacts are excluded from Git. Train the model and generate its metadata:

```powershell
python scripts/train_model.py
```

The script compares Logistic Regression, Decision Tree, and Random Forest on a
stratified validation split, selects by validation macro-F1, calibrates only
with training-partition folds, and evaluates the selected model once on the
held-out test set. It writes `artifacts/crop_model.joblib` and
`artifacts/model_metadata.json`.

Run the app:

```powershell
streamlit run app.py
```

Open `notebooks/crop_model_workflow.ipynb` in JupyterLab to follow the same
data-audit, exploration, split, comparison, calibration, evaluation, and
interpretation workflow interactively.

### Configuration

Open-Meteo's free API does not need a key. The default request timeout is eight
seconds. To override it, either set `WEATHER_TIMEOUT_SECONDS` in the
environment or copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and edit the timeout. The real secrets file is
ignored by Git. No credentials are required by this project.

## Checks

Run the focused tests with:

```powershell
python -m unittest discover -s tests -v
```

Tests mock weather requests and do not need a live API connection. Model
training and notebook execution require the downloaded CSV. No model accuracy
is claimed here; use the generated held-out evaluation results, and remember
that good performance on this dataset does not establish real-world reliability.
