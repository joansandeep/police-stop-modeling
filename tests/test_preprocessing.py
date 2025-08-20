import pandas as pd
from utils.preprocessing import basic_preprocessing, encode_for_xgb_mlp, encode_for_tabnet

def test_preprocessing():
    data = {
        'subject_age': [25, None, 40],
        'district': ['A', 'B', None],
        'subject_race': ['white', 'black', None],
        'subject_sex': ['male', 'female', None],
        'reason_for_stop': ['violation', None, 'other'],
        'arrest_made': [True, False, True]
    }
    df = pd.DataFrame(data)
    X, y = basic_preprocessing(df, 'arrest_made')
    assert X.isnull().sum().sum() == 0, "Missing values remain after preprocessing"
    X_enc = encode_for_xgb_mlp(X)
    assert X_enc.shape[0] == len(y), "Mismatch in features and targets"
    X_tab, cat_idxs, cat_dims = encode_for_tabnet(X)
    assert len(cat_idxs) > 0, "No categorical indices found for TabNet"
    print("Preprocessing tests passed.")

if __name__ == "__main__":
    test_preprocessing()
