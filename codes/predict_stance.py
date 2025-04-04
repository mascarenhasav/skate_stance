import pandas as pd

def predict_stance(model, new_person):
    """
    Prevê a stance (Regular ou Goofy) de um skatista com base em características de lateralidade.

    :param modelo: modelo de regressão logística treinado
    :param nova_pessoa: dict com características da nova pessoa
    :return: probabilidades de cada stance
    """
    new_person_df = pd.DataFrame([new_person])
    probabilidade = model.predict_proba(new_person_df)[0]
    return {
        "Goofy": f"{probabilidade[0]*100:.2f}%",
        "Regular": f"{probabilidade[1]*100:.2f}%"
    }
