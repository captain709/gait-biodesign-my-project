
import pickle
import matplotlib.pyplot as plt
# Use raw string (r"...") or double backslashes (\\)
file_path = r"C:\\Users\\Admin\\Desktop\\peam_biodis_gait\\result\\test_new_pipe\\img\\pytorch_RatAll_F1_10000.pkl"


with open(file_path, 'rb') as f:
    data = pickle.load(f)
data.show()
plt.show(block=False)
plt.pause(999999)
    
    
