import numpy as np
import cv2
import glob
from time import sleep
from datetime import datetime

TMP_FOLDER_PATH = "./tmp/"
MTX_PATH = TMP_FOLDER_PATH + "mtx.csv"
DIST_PATH = TMP_FOLDER_PATH + "dist.csv"

def main():
    calcCamera() 
    
    def calcCamera():
    square_size = 20.0      # 正方形のサイズ(mm)
    pattern_size = (9, 6)  # 模様のサイズ
    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 ) #チェスボード（X,Y,Z）座標の指定 (Z=0)
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    obj_points = []
    img_points = []
 
    for i in range(15):
    
        img
      
        found, corner = cv2.findChessboardCorners(img, pattern_size)
        # コーナーがあれば
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            corners2 = cv2.cornerSubPix(im, corner, (5,5), (-1,-1), term)
            # マークをつけて画像保存
            #img = cv2.drawChessboardCorners(im, pattern_size, corners2, found)
            #saveImgByTime(TMP_FOLDER_PATH, img)
            sleep(1)
        # コーナーがない場合のエラー処理
        if not found:
            print ('chessboard not found')
            continue
        img_points.append(corner.reshape(-1, 2))   #appendメソッド：リストの最後に因数のオブジェクトを追加
        obj_points.append(pattern_points)
 
    # 内部パラメータを計算
    rms, K, d, r, t = cv2.calibrateCamera(obj_points, img_points, (im.shape[1],im.shape[0]), None, None)

    # 計算結果を表示
    print ("RMS = ", rms)
    print ("K = \n", K)
    print ("d = ", d.ravel())

    # ファイル保存
    saveCalibrationFile(K, d, MTX_PATH, DIST_PATH)

# キャリブレーションCSVファイルを上書き保存する関数
def saveCalibrationFile(mtx, dist, mtx_path, dist_path):
    np.savetxt(mtx_path, mtx, delimiter =',',fmt="%0.14f")   #カメラ行列の保存
    np.savetxt(dist_path, dist, delimiter =',',fmt="%0.14f") #歪み係数の保存

if __name__ == '__main__':
    main()    

    
    
    
def calibrateImage():
    mtx, dist = loadCalibrationFile(MTX_PATH, DIST_PATH)

    for fn in glob.glob("./img/*"):
        img = cv2.imread(fn)
        resultImg = cv2.undistort(img, mtx, dist, None)
        saveImgByTime(SAVE_FOLDER_PATH, resultImg)
        sleep(1)

# キャリブレーションCSVファイルを読み込む関数
def loadCalibrationFile(mtx_path, dist_path):
    try:
        mtx = np.loadtxt(mtx_path, delimiter=',')
        dist = np.loadtxt(dist_path, delimiter=',')
    except Exception as e:
        raise e
    return mtx, dist
    
    
    
