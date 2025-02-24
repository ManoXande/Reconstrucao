import logging
import numpy as np
from pyautocad import Autocad, APoint
from sklearn.linear_model import RANSACRegressor

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('transformacao.log'), logging.StreamHandler()]
)

def initialize_autocad():
    """Inicializa a conexão com o AutoCAD."""
    try:
        acad = Autocad(create_if_not_exists=True)
        logging.info("Conexão com AutoCAD estabelecida.")
        return acad
    except Exception as e:
        logging.error(f"Falha na inicialização: {e}")
        return None

def select_survey_and_reference_data(acad):
    """Seleção em duas etapas: dados de levantamento e dados originais."""
    try:
        # Primeira seleção: COGO Points (dados de levantamento)
        logging.info("Selecione manualmente os COGO POINTS do levantamento")
        survey_set = None
        
        # Verificar se o conjunto de seleção já existe
        for i in range(acad.ActiveDocument.SelectionSets.Count):
            if acad.ActiveDocument.SelectionSets.Item(i).Name == "SURVEY_SET":
                survey_set = acad.ActiveDocument.SelectionSets.Item(i)
                break

        # Criar novo conjunto se necessário
        if survey_set is None:
            survey_set = acad.ActiveDocument.SelectionSets.Add("SURVEY_SET")
        else:
            survey_set.Clear()

        survey_set.SelectOnScreen()
        
        cogo_points = []
        for entity in survey_set:
            if entity.EntityName == 'AeccDbCogoPoint':
                cogo_points.append({
                    'Number': entity.Number,
                    'X': round(entity.Easting, 3),
                    'Y': round(entity.Northing, 3),
                    'Description': entity.RawDescription
                })
        
        # Segunda seleção: Polilinhas e Textos (dados originais)
        logging.info("Agora selecione POLILINHAS e TEXTOS de referência")
        ref_set = None
        
        # Verificar se o conjunto de seleção já existe
        for i in range(acad.ActiveDocument.SelectionSets.Count):
            if acad.ActiveDocument.SelectionSets.Item(i).Name == "REF_SET":
                ref_set = acad.ActiveDocument.SelectionSets.Item(i)
                break

        # Criar novo conjunto se necessário
        if ref_set is None:
            ref_set = acad.ActiveDocument.SelectionSets.Add("REF_SET")
        else:
            ref_set.Clear()

        logging.info("Selecione manualmente polilinhas e textos de referência")
        ref_set.SelectOnScreen()
        
        polylines, text_entities = [], []
        for entity in ref_set:
            if entity.EntityName == 'AcDbPolyline':
                vertices = [APoint(vertex) for vertex in entity.GetCoordinates()]
                polylines.append({'Vertices': vertices})
            elif entity.EntityName in ['AcDbText', 'AcDbMText']:
                text_entities.append({
                    'Position': APoint(entity.InsertionPoint),
                    'Content': entity.TextString.strip()
                })
                
        return cogo_points, polylines, text_entities
        
    except Exception as e:
        logging.error(f"Erro na extração de dados: {e}")
        return [], [], []

def map_points(cogo_points, polylines, text_entities):
    """Mapeia pontos reais para ideais usando textos como referência."""
    vertex_labels = {}
    ideal_points = []
    
    try:
        # Extrai vértices da primeira polilinha
        if polylines:
            ideal_poly = polylines[0]['Vertices']
            for i, vertex in enumerate(ideal_poly):
                vertex_labels[f"V{i+1}"] = (vertex.x, vertex.y)
        
        # Associa textos aos vértices
        for text in text_entities:
            for label, coords in vertex_labels.items():
                pt = APoint(coords[0], coords[1])
                if text['Position'].distance_to(pt) < 1.0:
                    vertex_labels[label] = (pt.x, pt.y)
        
        # Mapeia COGO points para vértices
        real_points, ideal_points_mapped = [], []
        for cogo in cogo_points:
            label = cogo['Description']
            if label in vertex_labels:
                real_points.append([cogo['X'], cogo['Y']])
                ideal_points_mapped.append(vertex_labels[label])
        
        return np.array(real_points), np.array(ideal_points_mapped)
        
    except Exception as e:
        logging.error(f"Erro no mapeamento: {e}")
        return np.array([]), np.array([])

def kabsch_transform(real_points, ideal_points):
    """Calcula transformação rígida usando algoritmo de Kabsch."""
    try:
        centroid_real = np.mean(real_points, axis=0)
        centroid_ideal = np.mean(ideal_points, axis=0)
        
        # Centralização dos pontos
        real_centered = real_points - centroid_real
        ideal_centered = ideal_points - centroid_ideal
        
        # Cálculo da matriz de covariância
        H = ideal_centered.T @ real_centered
        
        # Decomposição SVD
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        
        # Garante orientação correta
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
            
        # Calcula translação
        t = centroid_ideal - R @ centroid_real
        
        return R, t
        
    except Exception as e:
        logging.error(f"Erro na transformação de Kabsch: {e}")
        return None, None

def remove_outliers(real_points, ideal_points):
    """Remove outliers usando RANSAC."""
    try:
        model = RANSACRegressor(
            min_samples=2,
            residual_threshold=0.5,
            max_trials=100
        )
        model.fit(real_points, ideal_points)
        return real_points[model.inlier_mask_], ideal_points[model.inlier_mask_]
    except Exception as e:
        logging.error(f"Erro na remoção de outliers: {e}")
        return real_points, ideal_points

def delete_selection_sets(acad):
    """Remove os conjuntos de seleção criados."""
    try:
        for i in range(acad.ActiveDocument.SelectionSets.Count-1, -1, -1):
            ss = acad.ActiveDocument.SelectionSets.Item(i)
            if ss.Name in ["SURVEY_SET", "REF_SET"]:
                ss.Delete()
    except Exception as e:
        logging.error(f"Erro ao limpar conjuntos de seleção: {e}")

def draw_transformed_points(acad, real_points, R, t):
    """Desenha pontos transformados no AutoCAD."""
    try:
        for point in real_points:
            transformed = R @ point + t
            acad.model.AddPoint(APoint(transformed[0], transformed[1]))
        logging.info(f"{len(real_points)} pontos transformados desenhados.")
    except Exception as e:
        logging.error(f"Erro ao desenhar pontos: {e}")

def main():
    """Fluxo principal de execução."""
    acad = initialize_autocad()
    if not acad:
        return
    
    try:
        # Etapa 1: Extração de dados
        cogo_points, polylines, texts = select_survey_and_reference_data(acad)
        if not cogo_points:
            raise ValueError("Nenhum COGO point selecionado")
            
        # Etapa 2: Mapeamento de pontos
        real_points, ideal_points = map_points(cogo_points, polylines, texts)
        if len(real_points) < 2:
            raise ValueError("Mínimo de 2 pontos requeridos para transformação")
            
        # Etapa 3: Remoção de outliers
        real_clean, ideal_clean = remove_outliers(real_points, ideal_points)
        
        # Etapa 4: Transformação de Kabsch
        R, t = kabsch_transform(real_clean, ideal_clean)
        if R is None:
            return
            
        # Etapa 5: Visualização
        draw_transformed_points(acad, real_points, R, t)
        
        logging.info("Processo concluído com sucesso!")
        
    except Exception as e:
        logging.error(f"Erro no processo principal: {e}")
    finally:
        # Etapa 6: Limpeza dos conjuntos de seleção
        delete_selection_sets(acad)

if __name__ == "__main__":
    main()
