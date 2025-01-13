# Specify targets and features
targets = ['out:Leftover inventory[m2]', 'out:Displacement Average[cm]', 'out:Uncovered Wall Area[m2]', 'out:Average Mapping Tolerance']
features = ['in:Inventory', 'in:Wall Length', 'in:Wall Height', 'in:Door', 'in:Curve Frequency', 'in:Curve Amplitude'] 

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(combined_df[features], combined_df[targets], test_size=0.2, random_state=42)

# Initialize the PowerTransformer
scaler_yeo = PowerTransformer(method='yeo-johnson', standardize=False) 

# Fit and transform the target data
Y_train_trans = scaler_yeo.fit_transform(Y_train)

# Initialize StandardScaler
scaler_features = StandardScaler()
scaler_targets = StandardScaler()

# Fit and transform features
X_train_scaled = scaler_features.fit_transform(X_train)
Y_train_scaled = scaler_targets.fit_transform(Y_train_trans)

# Save the scaler for later use
joblib.dump(scaler_yeo, '/modelscaler_yeo.pkl')
joblib.dump(scaler_features, '/model/scaler_features.pkl')
joblib.dump(scaler_targets, '/model/scaler_targets.pkl')

# Initialize callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Define model architecture with Batch Normalization and L2 Regularization
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(Y_train.shape[1], activation='linear')  # Assuming a regression problem
])

# Compile model
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001)) #log spaced 0.01 up to 0.00001

# Fit model with callbacks
history = model.fit(X_train_scaled, Y_train_scaled, validation_split=0.1, epochs=100, batch_size=32, verbose=1,
                    callbacks=[early_stopping, reduce_lr])

# Save your model
model.save('/model/model.keras')

print('Model saved.')