import pandas as pd
import os

# Save Path
save_directory = 'D:/TUoS/Year_3/After Week_0/EEE380/2. RF_NSL/dataset/transformed'

# Save .CSV file path
train_csv_path = os.path.join(save_directory, 'KDDTrain+.csv')
test_csv_path = os.path.join(save_directory, 'KDDTest+.csv')

# Load KDDTrain+ file
train_file_path = 'D:/TUoS/Year_3/After Week_0/EEE380/2. RF_NSL/dataset/KDDTrain+.txt'
column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
                "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
                "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
                "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

# Load txt. and trans it to DataFrame
df_train = pd.read_csv(train_file_path, header=None, names=column_names)

# Save as .CSV file
df_train.to_csv(train_csv_path, index=False)
print("Training data has been successfully converted and saved in CSV format!")

# Trans Test file
test_file_path = 'D:/TUoS/Year_3/After Week_0/EEE380/2. RF_NSL/dataset/KDDTest+.txt'
df_test = pd.read_csv(test_file_path, header=None, names=column_names)
df_test.to_csv(test_csv_path, index=False)
print("Testing data has been successfully converted and saved in CSV format!")
