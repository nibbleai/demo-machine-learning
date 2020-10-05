# Machine Learning Demo App

A simple Machine Learning project used for educational purposes. **It is by no means a production-ready project.**

This project hosts 4 main systems:
 
- `train` for feature generation and model fitting
- `predict` for prediction
- `serve` to expose the prediction service via an HTTP REST API
- `deploy` to deploy a new model "in production" (_ie_ accessible by the `predict` system)


## Usage

**Python > 3.6 is required.** Install dependencies by running

```
$ pip install -r requirements.txt
```

If you run into import trouble, you can install the `src` package:

```bash
# Be sure to have a dedicated Python environment before...
$ pip install -e .
```

The application is controlled via a CLI. From the root directory, run:

```
$ python -m src.main --help
```

Here is the default documentation:

```
usage: python -m src.main [-h] [-f] [-t] [-hp] [-p] [-s] [-d] [--input INPUT]
                          [--disable-cache]

optional arguments:
  -h, --help          show this help message and exit
  -f, --features      Generate features
  -t, --train         Train the model with training data.
  -hp, --hyperopt     Train the model using hyperparameter optimization (only
                      valid with '--train')
  -p, --predict       Make a prediciton on sample data. ('--input' required)
  -s, --serve         Run a webserver locally to enable access to the
                      prediction service via HTTP requests.
  -d, --deploy-model  Deploy a serialized model to S3 for using in production.
  --input INPUT       A JSON-formatted string to use as feed for the predict
                      system.
  --disable-cache     Disable the caching system used to improve I/O
                      performance.


```

To run any service, you need an `.env` file at the root level, containing 4 variables:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET_NAME=...
USERNAME=...
```


## Data

Upon request, we will give you access to a CSV file containing the raw data on which this project is based. This CSV needs to be located in a `data` directory, at the root level.


## Content

In addition to the 4 systems listed above, some important modules are shared by all packages.

* `config.py` stores any global variable/constants 

* `aws.py` contains wrappers around `boto3` to interface with Cloud assets (namely S3 storage)


## License

MIT License

Copyright (c) 2020 nibble.ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
