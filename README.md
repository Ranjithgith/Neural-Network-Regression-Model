# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

<img width="842" height="665" alt="image" src="https://github.com/user-attachments/assets/df99db89-d0f4-42d2-9108-770c212ba05a" />



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
### Register Number: 212224240131
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1, 16)
        self.hidden2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return x

R_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(R_brain.parameters(), lr=0.01)


def train_model(R_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = R_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        R_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
## Dataset Information

<img width="191" height="529" alt="image" src="https://github.com/user-attachments/assets/a6b12664-78eb-497a-a179-a3bfc7968c46" />



## OUTPUT

### Training Loss Vs Iteration Plot

<img width="1243" height="698" alt="image" src="https://github.com/user-attachments/assets/d0026594-a7c0-4199-8f56-237ab72c6292" />




### New Sample Data Prediction

```
X_n1_1 = torch.tensor([[7]], dtype=torch.float32)
prediction = R_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction:.4f}')
```

<img width="603" height="54" alt="image" src="https://github.com/user-attachments/assets/a74c6777-e8bd-48b5-a617-4be044595201" />


## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
