
import lightgbm as lgb
import timeit as t
import logging
from duplicate_model import DuplicateModel

model_path = 'model.pkl'
questions_path = 'questions.tsv'
logger = logging.getLogger("model_driver")

def create_scoring_func():
    """ Initialize Model Object 
    """   
    start = t.default_timer()
    DM = DuplicateModel(model_path, questions_path)
    end = t.default_timer()
    
    loadTimeMsg = "Model object loading time: {0} ms".format(round((end-start)*1000, 2))
    logger.info(loadTimeMsg)
    
    def call_model(text):
        preds = DM.score(text)  
        return preds
    
    return call_model

def get_model_api():
    logger = logging.getLogger("model_driver")
    scoring_func = create_scoring_func()
    
    def process_and_score(inputString):
        """ Classify the input using the loaded model
        """
        start = t.default_timer()
        responses = scoring_func(inputString)
        end = t.default_timer()
        
        logger.info("Predictions: {0}".format(responses))
        logger.info("Predictions took {0} ms".format(round((end-start)*1000, 2)))
        return (responses, "Computed in {0} ms".format(round((end-start)*1000, 2)))
    return process_and_score

def version():
    return lgb.__version__