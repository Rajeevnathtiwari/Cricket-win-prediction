
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import tkinter as tk


# Load ODI batting dataset
odi_batting_path = r"E:\PROJECTS\MINOR 2\ANN based cricket pridiction\Batting_odi1.csv"
odi_batting_data = pd.read_csv(odi_batting_path)

#Load ODI bowling dataset
odi_bowling_path = r"E:\PROJECTS\MINOR 2\ANN based cricket pridiction\Bowling_ODI1.csv"
odi_bowling_data = pd.read_csv(odi_bowling_path)

# Extract common numeric columns for batting data
batting_numeric_columns = ['Runs', 'HS', 'SR']

# Extract common numeric columns for bowling data
bowling_numeric_columns = ['Balls', 'Wkts', 'Econ']

# Standardize features for ODI batting data
scaler_batting = StandardScaler()
odi_batting_data_scaled = scaler_batting.fit_transform(odi_batting_data[batting_numeric_columns])

# Standardize features for ODI bowling data
scaler_bowling = StandardScaler()
odi_bowling_data_scaled = scaler_bowling.fit_transform(odi_bowling_data[bowling_numeric_columns])

# Load labels for ODI matches
odi_labels_path = r"E:\PROJECTS\MINOR 2\ANN based cricket pridiction\ODI_label.csv"
odi_labels = pd.read_csv(odi_labels_path)

# Define criteria to split labels into subsets for batting and bowling data
# Assuming the labels are ordered in the same way as the data
batting_labels = odi_labels.iloc[:len(odi_batting_data)]
bowling_labels = odi_labels.iloc[:len(odi_bowling_data)]

# Encode labels to have 'win' and 'lose' instead of 0 and 1 for batting data
label_encoder_batting = LabelEncoder()
odi_labels_encoded_batting = label_encoder_batting.fit_transform(batting_labels.values.ravel())

# Encode labels to have 'win' and 'lose' instead of 0 and 1 for bowling data
label_encoder_bowling = LabelEncoder()
odi_labels_encoded_bowling = label_encoder_bowling.fit_transform(bowling_labels.values.ravel())

# Create and train the feedforward neural network classifier for ODI matches (batting data)
odi_classifier_batting = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=42)
odi_classifier_batting.fit(odi_batting_data_scaled, odi_labels_encoded_batting)

# Create and train the feedforward neural network classifier for ODI matches (bowling data)
odi_classifier_bowling = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=42)
odi_classifier_bowling.fit(odi_bowling_data_scaled, odi_labels_encoded_bowling)

def get_player_data(player_names, data_type):
    if data_type == 'batting':
        dataset = odi_batting_data
    elif data_type == 'bowling':
        dataset = odi_bowling_data

    player_data = dataset[dataset['Player'].isin(player_names)]
    return player_data[batting_numeric_columns if data_type == 'batting' else bowling_numeric_columns]

def predict_match_outcome(team_a_batting, team_a_bowling, team_b_batting, team_b_bowling):
    # Calculate average performance scores for Team A batting and bowling
    avg_score_team_a_batting = team_a_batting.mean().mean()
    avg_score_team_a_bowling = team_a_bowling.mean().mean()

    # Calculate average performance scores for Team B batting and bowling
    avg_score_team_b_batting = team_b_batting.mean().mean()
    avg_score_team_b_bowling = team_b_bowling.mean().mean()

    # Determine the winning team based on average performance scores
    if avg_score_team_a_batting + avg_score_team_a_bowling > avg_score_team_b_batting + avg_score_team_b_bowling:
        summary = "Team A wins because they performed better."
    elif avg_score_team_b_batting + avg_score_team_b_bowling > avg_score_team_a_batting + avg_score_team_a_bowling:
        summary = "Team B wins because they performed better."
    else:
        summary = "It's a tie! Both teams performed equally well."

    return summary

def get_selected_players(player_type, num_players):
    root = tk.Tk()
    root.title(f"Select {player_type} Players")

    player_options = odi_batting_data['Player'] if player_type == 'Batting' else odi_bowling_data['Player']
    player_vars = [tk.IntVar() for _ in player_options]

    frame = tk.Frame(root)
    frame.pack(fill='both', expand=True)

    search_frame = tk.Frame(frame)
    search_frame.pack(fill='x', padx=10, pady=5)

    search_label = tk.Label(search_frame, text="Search:")
    search_label.pack(side='left')

    search_entry = tk.Entry(search_frame)
    search_entry.pack(side='left', expand=True, fill='x', padx=(5, 0))

    canvas = tk.Canvas(frame)
    scrollbar = tk.Scrollbar(frame, orient='vertical', command=canvas.yview)
    scrollbar.pack(side='right', fill='y')

    inner_frame = tk.Frame(canvas)

    canvas.create_window((0, 0), window=inner_frame, anchor='nw')
    canvas.config(yscrollcommand=scrollbar.set)

    player_checkboxes = []

    def filter_players(event=None):
        search_text = search_entry.get().lower()
        for widget in inner_frame.winfo_children():
            widget.destroy()
        for player, var in zip(player_options, player_vars):
            if search_text in player.lower():
                checkbox = tk.Checkbutton(inner_frame, text=player, variable=var)
                checkbox.pack(anchor='w', padx=10, pady=5)
                player_checkboxes.append(checkbox)

    search_entry.bind('<KeyRelease>', filter_players)

    filter_players()

    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox('all'))

    inner_frame.bind('<Configure>', on_configure)

    canvas.pack(side='left', fill='both', expand=True)

    done_button = tk.Button(root, text="Done", command=root.destroy)
    done_button.pack(pady=10)

    root.mainloop()

    selected_players = [player for player, var in zip(player_options, player_vars) if var.get() == 1]
    return selected_players


# Get player names for Team A
team_a_batting_players = get_selected_players("Batting", 6)
team_a_bowling_players = get_selected_players("Bowling", 5)

# Get player names for Team B
team_b_batting_players = get_selected_players("Batting", 6)
team_b_bowling_players = get_selected_players("Bowling", 5)

# Retrieve data for Team A and Team B batting and bowling
team_a_batting = get_player_data(team_a_batting_players, 'batting')
team_a_bowling = get_player_data(team_a_bowling_players, 'bowling')
team_b_batting = get_player_data(team_b_batting_players, 'batting')
team_b_bowling = get_player_data(team_b_bowling_players, 'bowling')

# Call the predict_match_outcome function with the retrieved DataFrames
match_outcome = predict_match_outcome(team_a_batting, team_a_bowling, team_b_batting, team_b_bowling)
print("Match outcome:", match_outcome)

print("---------------------------------------------")

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# For batting data
X_train_batting, X_test_batting, y_train_batting, y_test_batting = train_test_split(
    odi_batting_data_scaled, odi_labels_encoded_batting, test_size=0.5, random_state=42)

# For bowling data
X_train_bowling, X_test_bowling, y_train_bowling, y_test_bowling = train_test_split(
    odi_bowling_data_scaled, odi_labels_encoded_bowling, test_size=0.5, random_state=42)
odi_classifier_batting.fit(X_train_batting, y_train_batting)
odi_classifier_bowling.fit(X_train_bowling, y_train_bowling)
# Predictions for batting data
y_pred_batting = odi_classifier_batting.predict(X_test_batting)
conf_matrix_batting = confusion_matrix(y_test_batting, y_pred_batting)
accuracy_batting = accuracy_score(y_test_batting, y_pred_batting)

# Predictions for bowling data
y_pred_bowling = odi_classifier_bowling.predict(X_test_bowling)
conf_matrix_bowling = confusion_matrix(y_test_bowling, y_pred_bowling)
accuracy_bowling = accuracy_score(y_test_bowling, y_pred_bowling)

# Display the results
print("Accuracy for Batting Model: {:.1f}%".format(accuracy_batting * 100))
print("Confusion Matrix for Batting Model:\n", conf_matrix_batting)

print("Accuracy for Bowling Model: {:.1f}%".format(accuracy_bowling * 100))
print("Confusion Matrix for Bowling Model:\n", conf_matrix_bowling)

# Create confusion matrix displays using the updated method
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# For batting data
ConfusionMatrixDisplay.from_predictions(y_test_batting, y_pred_batting, ax=ax[0], cmap='Blues')
ax[0].set_title('Confusion Matrix for Batting Model')

# For bowling data
ConfusionMatrixDisplay.from_predictions(y_test_bowling, y_pred_bowling, ax=ax[1], cmap='Blues')
ax[1].set_title('Confusion Matrix for Bowling Model')

plt.show()
print("---------------------")

plt.show()