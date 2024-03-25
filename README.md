# Predictive Compliance Monitoring

## Description
Code accompanying the paper Quantifying The Degree Of Process Compliance: A Predictive Compliance Monitoring Approach

## Getting started

Create a virtual environment
```
python3 -m venv venv
```

Activate your virtual environment
```
source venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

Process training and test data sets
```
python3 data_processing.py
```

Next-attributes-based PCM
```
python3 next_predictions.py
```

Continuation-based PCM
```
python3 pcm.py
```

## License
LGPL-3.0 license
