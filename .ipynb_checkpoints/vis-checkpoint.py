import json
import scipy.io as sio # Scipy input and output

def  load_dataset(dataset):
    """load dataset parameters from config.json"""
    
    with open('./config.json') as f:
        config = json.loads(f.read())
        params = config[dataset]
        data = sio.loadmat(params['img_path'])[params['img']]
        labels = sio.loadmat(params['gt_path'])[params['gt']]
        num_classes = params['num_classes']
        target_names = params['target_names']
        
    return data,labels,num_classes,target_names


dataset = "PaviaUSh" # Indian_pines or PaviaU or or Salinas  . check config.json
X, y , num_classes , target_names = load_dataset(dataset)
print("Initial {}".format(X.shape))
