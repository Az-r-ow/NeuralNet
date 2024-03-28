# Train and Predict MNIST

In this example we train the NeuralNet module to predict handwritten digits.

## Setup :

Create a virtual environment with this command :

```bash
python3 -m venv venv
```

Then activate the virtual env with the command :

```bash
source venv/bin/activate
```

On Windows :

```bash
venv\Scripts\activate
```

Once the virtual environment activated, install the python dependencies.

```bash
python -m pip install -r requirements.txt
```

Finally, create the dataset folder :

```bash
mkdir dataset
```

In the end to deactivate the virtual env, execute this command :

```bash
deactivate
```

## Getting Started

First I would recommend checking out the `main.py` file, it's the entrypoint of the example.

It will :

- Download the MNIST handwritten database (as .npz)
- Initialize a `Network` (Neural Network duh!)
- Prepare and train the `Network` on the data
- Save the trained `Network`'s parameters in a `.bin` file
- Test the `Network`'s accuracy on unseen data

Run it with

```bash
python main.py
```

| ⚠️ It might take a while depending on your machine cpu |
| ------------------------------------------------------ |

After training and saving the network's parameters in the file, you can now run the interactive example with :

```bash
python guess_it.py
```

Which will launch a window on which you can draw a number and let the model that you previously trained guess the number.

- **Left** click to **draw**
- **Right** click to **erase**
