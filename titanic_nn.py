import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Function to preprocess the data
def preprocess_data(data, is_training=True):
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = data[features].copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X[['Age', 'Fare']] = imputer.fit_transform(X[['Age', 'Fare']])
    
    # Encode categorical variables
    le = LabelEncoder()
    X['Sex'] = le.fit_transform(X['Sex'])
    X['Embarked'] = le.fit_transform(X['Embarked'].fillna('S'))
    
    # Scale numerical features
    scaler = StandardScaler()
    X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])
    
    return X

# Preprocess training data
X_train = preprocess_data(train_data)
y_train = train_data['Survived']

# Preprocess test data
X_test = preprocess_data(test_data)

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(7,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss',
                              patience=10,
                              restore_best_weights=True)

model_checkpoint = ModelCheckpoint('best_model.h5',
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 mode='max',
                                 verbose=1)

# Train the model
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Evaluate on validation set
val_predictions = (model.predict(X_val) > 0.5).astype(int)
val_accuracy = accuracy_score(y_val, val_predictions)
print("\nValidation Accuracy:", val_accuracy)
print("\nClassification Report:")
print(classification_report(y_val, val_predictions))

# Plot confusion matrix
cm = confusion_matrix(y_val, val_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Make predictions on test data
predictions = (model.predict(X_test) > 0.5).astype(int)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions.flatten()
})

submission.to_csv('submission.csv', index=False)
print("\nPredictions saved to submission.csv")
print("Model saved as best_model.h5")
print("Training history plot saved as training_history.png")
print("Confusion matrix plot saved as confusion_matrix.png") 