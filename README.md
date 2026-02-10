# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The class NeuralNet inherits from nn.Module, which is the base class for all neural networks in PyTorch.

In the constructor init, layers and activation functions are defined.

The first layer self.n1=nn.Linear(1,10) takes one input feature and maps it to 10 neurons.

The second layer self.n2=nn.Linear(10,20) processes the 12 outputs and maps them to 20 neurons.

The third layer self.n3=nn.Linear(20,1) reduces the 14 features back to a single output.

The activation function self.relu=nn.ReLU() introduces non-linearity, helping the network learn complex patterns.

A history dictionary is initialized to store the loss values during training for performance tracking.

The forward function defines how input data flows through the network layers.

Input x is first passed through n1 and activated by ReLU, then through n2 with ReLU again.

Finally, the processed data passes through n3 to produce the output, which is returned.

## Neural Network Model

<img width="705" height="741" alt="Screenshot 2026-02-10 151611" src="https://github.com/user-attachments/assets/e2133b64-a19a-43cd-a503-2964c0084602" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: RANJIT R
### Register Number:212224240131
```
class Neuralnet(nn.Module):
   def __init__(self):
        super().__init__()
        self.n1=nn.Linear(1,10)
        self.n2=nn.Linear(10,20)
        self.n3=nn.Linear(20,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}
   def forward(self,x):
        x=self.relu(self.n1(x))
        x=self.relu(self.n2(x))
        x=self.n3(x)
        return x


## Initialize the Model, Loss Function, and Optimizer
deva_brain=Neuralnet()
criteria=nn.MSELoss()
optimizer=optim.RMSprop(deva_brain.parameters(),lr=0.001)

def train_model(deva_brain,x_train,y_train,criteria,optmizer,epochs=4000):
    for i in range(epochs):
        optimizer.zero_grad()
        loss=criteria(deva_brain(x_train),y_train)
        loss.backward()
        optimizer.step()
        
        deva_brain.history['loss'].append(loss.item())
        if i%200==0:
            print(f"Epoch [{i}/epochs], loss: {loss.item():.6f}")

```

## Dataset Information

<img width="308" height="555" alt="image" src="https://github.com/user-attachments/assets/9493f983-ae5f-467a-9efc-6f6dc992be37" />


## OUTPUT

<img width="424" height="442" alt="Screenshot 2026-02-10 151215" src="https://github.com/user-attachments/assets/156e2d7e-f1b4-4d97-aa8c-dbd25b580e41" />


### Training Loss Vs Iteration Plot

<img width="754" height="579" alt="Screenshot 2026-02-10 151231" src="https://github.com/user-attachments/assets/df284cfe-d9d5-4c59-b910-666a95f919b6" />


### New Sample Data Prediction


<img width="865" height="139" alt="Screenshot 2026-02-10 151223" src="https://github.com/user-attachments/assets/f9fd6f8d-5e99-4de4-9d59-adac17858df4" />


## RESULT

Successfully executed the code to develop a neural network regression model.
