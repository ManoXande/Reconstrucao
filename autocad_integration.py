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
        logging.error(f"Falha na inicialização: {e}", exc_info=True)
        return None

def select_survey_and_reference_data(acad):
    """Seleção em duas etapas: dados de levantamento e dados originais."""
    try:
        # Primeira seleção: COGO Points (dados de levantamento)
        logging.info("Selecione manualmente os COGO POINTS do levantamento")
        survey_set = None
        
        # Verificar/Criar conjunto de seleção
        for i in range(acad.ActiveDocument.SelectionSets.Count):
            if acad.ActiveDocument.SelectionSets.Item(i).Name == "SURVEY_SET":
                survey_set = acad.ActiveDocument.SelectionSets.Item(i)
                break

        if not survey_set:
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
        
        # Segunda seleção: Polilinhas e Textos
        logging.info("Agora selecione POLILINHAS e TEXTOS de referência")
        ref_set = None
        
        # Verificar/Criar conjunto de seleção
        for i in range(acad.ActiveDocument.SelectionSets.Count):
            if acad.ActiveDocument.SelectionSets.Item(i).Name == "REF_SET":
                ref_set = acad.ActiveDocument.SelectionSets.Item(i)
                break

        if not ref_set:
            ref_set = acad.ActiveDocument.SelectionSets.Add("REF_SET")
        else:
            ref_set.Clear()

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
        logging.error(f"Erro na extração de dados: {e}", exc_info=True)
        return [], [], []

def map_points(cogo_points, polylines, text_entities):
    """Mapeia pontos reais para ideais usando textos como referência."""
    vertex_labels = {}
    
    try:
        if not polylines:
            raise ValueError("Nenhuma polilinha de referência selecionada")
            
        if not text_entities:
            logging.warning("Nenhum texto de referência encontrado - usando vértices brutos")

        # Extrair e rotular vértices
        for poly_idx, poly in enumerate(polylines):
            vertices = poly['Vertices']
            for vert_idx, vertex in enumerate(vertices):
                label = f"V{poly_idx+1}-{vert_idx+1}"
                vertex_labels[label] = (vertex.x, vertex.y)

        # Associar textos aos vértices mais próximos
        if text_entities:
            for text in text_entities:
                text_pos = text['Position']
                closest = min(
                    vertex_labels.items(),
                    key=lambda item: text_pos.distance_to(APoint(item[1][0], item[1][1])),
                    default=None
                )
                
                if closest and text_pos.distance_to(APoint(closest[1][0], closest[1][1])) < 5.0:
                    vertex_labels[closest[0]] = (text_pos.x, text_pos.y)
                    logging.info(f"Texto '{text['Content']}' associado ao vértice {closest[0]}")
                else:
                    logging.warning(f"Texto '{text['Content']}' não associado a nenhum vértice")

        # Mapear pontos COGO
        real_points, ideal_points = [], []
        missing_labels = set()
        
        for cogo in cogo_points:
            label = cogo['Description'].strip().upper()
            if not label:
                logging.error(f"COGO point {cogo['Number']} sem descrição")
                continue
                
            if label in vertex_labels:
                real_points.append([cogo['X'], cogo['Y']])
                ideal_points.append(vertex_labels[label])
            else:
                missing_labels.add(label)
                logging.error(f"Descrição '{label}' não encontrada para COGO point {cogo['Number']}")

        if not real_points:
            raise ValueError("Nenhum ponto válido mapeado")
            
        logging.info(f"Pontos mapeados: {len(real_points)}")
        logging.info(f"Descrições faltantes: {', '.join(missing_labels) if missing_labels else 'Nenhuma'}")
        
        return np.array(real_points), np.array(ideal_points)
        
    except Exception as e:
        logging.error(f"Erro no mapeamento: {str(e)}", exc_info=True)
        return np.array([]), np.array([])

def kabsch_transform(real_points, ideal_points):
    """Calcula transformação rígida usando algoritmo de Kabsch."""
    try:
        if real_points.size == 0 or ideal_points.size == 0:
            raise ValueError("Pontos de entrada vazios para cálculo da transformação")
            
        centroid_real = np.mean(real_points, axis=0)
        centroid_ideal = np.mean(ideal_points, axis=0)
        
        H = (ideal_points - centroid_ideal).T @ (real_points - centroid_real)
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
            
        t = centroid_ideal - R @ centroid_real
        
        return R, t
    except Exception as e:
        logging.error(f"Falha na transformação de Kabsch: {str(e)}", exc_info=True)
        return None, None

def remove_outliers(real_points, ideal_points):
    """Remove outliers usando RANSAC."""
    try:
        if real_points.size == 0 or ideal_points.size == 0:
            logging.warning("Dados insuficientes para filtragem de outliers")
            return real_points, ideal_points
            
        model = RANSACRegressor(
            min_samples=2,
            residual_threshold=0.5,
            max_trials=100
        )
        model.fit(real_points, ideal_points)
        return real_points[model.inlier_mask_], ideal_points[model.inlier_mask_]
    except Exception as e:
        logging.error(f"Erro na filtragem de outliers: {str(e)}", exc_info=True)
        return real_points, ideal_points

def main():
    """Fluxo principal de execução."""
    acad = initialize_autocad()
    if not acad:
        return
    
    try:
        cogo_points, polylines, texts = select_survey_and_reference_data(acad)
        real_points, ideal_points = map_points(cogo_points, polylines, texts)
        
        if len(real_points) < 2:
            raise ValueError("Mínimo de 2 pontos requeridos para transformação")
            
        real_clean, ideal_clean = remove_outliers(real_points, ideal_points)
        R, t = kabsch_transform(real_clean, ideal_clean)
        
        if R is not None:
            for point in real_points:
                transformed = R @ point + t
                acad.model.AddPoint(APoint(transformed[0], transformed[1]))
            logging.info(f"{len(real_points)} pontos transformados desenhados")
            
    except Exception as e:
        logging.error(f"Erro no processo principal: {str(e)}", exc_info=True)
    finally:
        try:
            for i in range(acad.ActiveDocument.SelectionSets.Count-1, -1, -1):
                ss = acad.ActiveDocument.SelectionSets.Item(i)
                if ss.Name in ["SURVEY_SET", "REF_SET"]:
                    ss.Delete()
        except Exception as e:
            logging.error(f"Erro na limpeza: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
