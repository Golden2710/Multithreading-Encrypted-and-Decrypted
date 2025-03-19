import numpy as np
import threading 
from math import acos, sin
import cv2
import struct
from typing import Tuple,List
import time
import math
start_time = time.time()

#Constants from paper
NUMBER_OF_THREADS = 8
CONFUSION_DIFFUSION_ROUNDS = 2
CONFUSION_SEED_UPPER_BOUND = 10000
CONFUSION_SEED_LOWER_BOUND = 3000
PRE_ITERATIONS = 1000
BYTES_RESERVED = 6
PI = acos(-1)

class PLCM:
    def __init__(self,control_param: float, init_condition:float):
        if control_param >= 0.5:
            control_param = 1- control_param
        self.p = control_param
        self.x = init_condition
        self._pre_iterate()
    
    def _pre_iterate(self):
        for _ in range(PRE_ITERATIONS):
            self.x = self._single_iterate(self.x)
    
    def _single_iterate(self, x:float) -> float:
        if 0 <= x < self.p:
            return x / self.p
        elif self.p <= x <= 0.5:
            return (x - self.p) / (0.5 - self.p)
        else:
            return self._single_iterate(1.0 -x)
        
    def _double_to_bytes(self, value: float) -> bytes:
        return struct.pack('d',value)[2: 2+ BYTES_RESERVED]   # 8byte = 64 bit
    
    def iterate_and_get_bytes(self, iterations: int ) -> Tuple[float, List[bytes]]:
        x = self.x 
        byte_list = []

        for _ in range(iterations):
            x = self._single_iterate(x)
            byte_list.append(self._double_to_bytes(x))
        self.x = x
        return x, byte_list
    
class AssistantThread(threading.Thread):
    def __init__(self, thread_idx: int, image_shape: Tuple[int,int], init_params: Tuple[Tuple[float, float], ...],shared_data : dict, lock : threading.Lock):
        super().__init__()
        self.thread_idx = thread_idx
        self.height, self.width = image_shape
        self.rows_per_thread = self.height // NUMBER_OF_THREADS
        self.start_row = thread_idx * self.rows_per_thread
        self.end_row = self.start_row + self.rows_per_thread
        self.init_params = init_params
        self.plcm1 = PLCM(*init_params[0])
        self.plcm2 = PLCM(*init_params[1])

        self.frame_ready = threading.Event()
        self.frame_processed = threading.Event()
        self.lock = lock
        self.shared_data = shared_data

    def reset(self):
        """Reset PLCMs về trạng thái ban đầu"""
        self.plcm1 = PLCM(*self.init_params[0])
        self.plcm2 = PLCM(*self.init_params[1])
    def inverse_confusion_operation(self):
        with self.lock:
            temp_frame = self.shared_data['temp_frame']
            confusion_seed = self.shared_data['confusion_seed']
            result_frame = self.shared_data['confused_frame']
        
        for r in range(self.start_row, self.end_row):
            for c in range(self.width):
                #temp = round(confusion_seed * sin(2 * PI * r / self.height))
                #alpha = (r - c +math.floor(confusion_seed * sin(2 * PI * r / self.height))) % self.height
                #beta = (c-int(confusion_seed*sin(2 * PI * r / self.height))) % self.width
                for alpha in range(self.height):
                    for beta in range (self.width):
                        if r == (alpha + beta) % self.height and c == (beta + int(confusion_seed * sin(2 * PI * r / self.height))) % self.width:
                            with self.lock:
                                temp_frame[alpha, beta] = result_frame[r, c]
        with self.lock:
            self.shared_data['temp_frame']=temp_frame

    def generate_byte_sequence(self) -> np.ndarray:
        pixels = (self.end_row - self.start_row) * self.width
        iterations = (pixels + BYTES_RESERVED - 1) // BYTES_RESERVED
        _,bytes1 = self.plcm1.iterate_and_get_bytes(iterations)
        _,bytes2 = self.plcm2.iterate_and_get_bytes(iterations)

        # Convert bytes to numpy arrays and XOR
        arr1 = np.frombuffer(b''.join(bytes1), dtype=np.uint8)
        arr2 = np.frombuffer(b''.join(bytes2), dtype = np.uint8)     
        return np.bitwise_xor(arr1,arr2)[:pixels]    

    
    def inverse_diffusion_operation(self):
        byte_seq = self.generate_byte_sequence()
    
        with self.lock:
            temp_frame = np.zeros_like(self.shared_data['diffused_frame'])
            result_frame = self.shared_data['diffused_frame'].copy()
            diffusion_seed_val= int(self.shared_data['diffusion_seed'])

        seq_idx = len(byte_seq)-1
    
        # Duyệt ngược và tính toán lại giá trị gốc
        for i in reversed(range(self.start_row, self.end_row)):
            for j in reversed(range(self.width)):
                byte_val= byte_seq[seq_idx]
            
                if i == self.start_row and j == 0:
                    # Tính toán giá trị gốc từ pixel đã mã hóa và seed
                    temp_sum_decrypted= (result_frame[i,j].astype(np.int32) ^ diffusion_seed_val ^ byte_val)
                    original_pixel= (temp_sum_decrypted - byte_val) % 256
                    temp_frame[i,j]= original_pixel
            
                else:
                    #prev_i= i if j < self.width-1 else i-1
                    #prev_j= j-1 if j < self.width-1 else 0
                    prev_i = i if j > 0 else i-1
                    prev_j = j-1 if j > 0 else self.width-1

                    with self.lock:
                        prev_pixel_decrypted= result_frame[prev_i][prev_j]
                        temp_sum_decrypted= (result_frame[i,j].astype(np.int32)  ^ prev_pixel_decrypted ^ byte_val) 
                        original_pixel= (temp_sum_decrypted - byte_val) % 256 
                        temp_frame[i,j]= original_pixel
            
                seq_idx -=1
    
        with self.lock:
            self.shared_data['temp_frame'][self.start_row:self.end_row,:]= \
            temp_frame[self.start_row:self.end_row,:].copy()#'''


    def run(self):
        while True:
            #Inverse Diffusion phase
            self.frame_ready.wait()
            self.frame_ready.clear()
            self.inverse_diffusion_operation()
            self.frame_processed.set()

            #Inverse Diffusion phase
            self.frame_ready.wait()
            self.frame_ready.clear()
            self.inverse_confusion_operation()
            self.frame_processed.set()

class ImageEncryptionSystem:
    def __init__(self, image: np.ndarray):
        self.image = image.astype(np.uint8)
        self.height, self.width = image.shape[0], image.shape[1]
        self.lock = threading.Lock()

        np.random.seed()
        self.main_plcm = PLCM(control_param= 0.37, init_condition= 0.2)

        self.shared_data = {
            'temp_frame': np.zeros_like(image),
            'confused_frame': np.zeros_like(image),
            'diffused_frame': np.zeros_like(image),
            'confusion_seed': 0,
            'diffusion_seed': 0,
        }
        self.threads=self.__initialize_threads()
    
    def _generate_thread_parameters(self):
        params = []
        x = self.main_plcm.x
        
        for _ in range(NUMBER_OF_THREADS):
            x = self.main_plcm._single_iterate(x)
            p1 = x
            x = self.main_plcm._single_iterate(x)
            x1 = x
            x = self.main_plcm._single_iterate(x)
            p2 = x
            x = self.main_plcm._single_iterate(x)
            x2 = x

            params.append(((p1,x1),(p2,x2)))
        self.main_plcm.x = x
        return params
    
    def __initialize_threads(self):
        threads = []
        thread_params = self._generate_thread_parameters()

        for i in range(NUMBER_OF_THREADS):
            thread = AssistantThread(
                thread_idx= i, 
                image_shape=(self.height, self.width),
                init_params = thread_params[i],
                shared_data= self.shared_data,
                lock = self.lock
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)

        return threads
    
    def _reset_system(self):
        """Reset toàn bộ hệ thống về trạng thái ban đầu"""
        self.main_plcm = PLCM(0.37, 0.2)
        thread_params = self._generate_thread_parameters()
        
        for i, thread in enumerate(self.threads):
            thread.init_params = thread_params[i]
            thread.reset()

    def decrypt(self)->np.ndarray:
        with self.lock: 
            current_state=self.image.copy()
        
        # Xử lý các vòng theo thứ tự ngược với mã hóa  
        for _ in reversed(range(CONFUSION_DIFFUSION_ROUNDS)):
            self._reset_system()
            # Phase 1: Inverse Diffusion  
            x=self.main_plcm._single_iterate(self.main_plcm.x)
            
            with self.lock:
                self.shared_data.update({
                    'diffusion_seed':int(x*256)&0xFF,
                    'diffused_frame':current_state.copy(),
                    'temp_frame':current_state.copy()
                })
            
            for t in self.threads:t.frame_ready.set()
            for t in self.threads:t.frame_processed.wait();t.frame_processed.clear()

            with self.lock:
                current_state=self.shared_data['temp_frame'].copy()
            
            # Phase 2: Inverse Confusion
            x=self.main_plcm._single_iterate(self.main_plcm.x)

            seed=int(abs(x)*CONFUSION_SEED_UPPER_BOUND)%\
                (CONFUSION_SEED_UPPER_BOUND-CONFUSION_SEED_LOWER_BOUND)+\
                CONFUSION_SEED_LOWER_BOUND
            
            with self.lock:
                self.shared_data.update({
                    'confusion_seed':seed,
                    'confused_frame':current_state.copy(),
                    'temp_frame':current_state.copy()
                })
            # Kích hoạt và chờ các thread hoàn thành Inverse Confusion  
            for t in self.threads:t.frame_ready.set()
            for t in self.threads:t.frame_processed.wait();t.frame_processed.clear()

            with self.lock:
                current_state=self.shared_data['temp_frame'].copy()
        return current_state.copy()

def decrypt_image(image) -> Tuple[np.ndarray, np.ndarray]:
       
        if image is None:
            raise ValueError("Could not read image")
    
        height, width  = image.shape[0],image.shape[1]
        if height % NUMBER_OF_THREADS != 0:
           pad_height = ((height + NUMBER_OF_THREADS - 1) // NUMBER_OF_THREADS) * NUMBER_OF_THREADS
           image = np.pad(image, ((0, pad_height - height), (0, 0)), mode='reflect')
    
        system = ImageEncryptionSystem(image)
        decrypted_padded = system.decrypt()    
        decrypted = decrypted_padded[:width, :]
        return image, decrypted
    
def decrypt_image_BGR(image):
        
        B, G, R =cv2.split(image)
        L=[B,G,R]
        DecryptedBGR = []
        for x in L:
            if x is None:
                raise ValueError("Could not read image")
    
            height,width = x.shape[0], x.shape[1]
            if height % NUMBER_OF_THREADS != 0:
                pad_height = ((height + NUMBER_OF_THREADS - 1) // NUMBER_OF_THREADS) * NUMBER_OF_THREADS
                x = np.pad(x, ((0, pad_height - height), (0, 0)), mode='reflect')
            system = ImageEncryptionSystem(x)
            decrypted_padded = system.decrypt()
            decrypted = decrypted_padded[:width, :]
            DecryptedBGR.append(decrypted)
        encrytped_image_color = cv2.merge((DecryptedBGR[0],DecryptedBGR[1],DecryptedBGR[2]))
        return encrytped_image_color

def main():
    img_path = 'encrypted_image_2round_BGR.png'
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    cv2.imshow('img',img)
    decrypted_image =decrypt_image_BGR(img)
    cv2.imshow('decrypted',decrypted_image)
    cv2.imwrite('original_image.png',decrypted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    end_time =time.time()
    process_time = end_time-start_time
    print(process_time)
    print(decrypted_image.shape)

if __name__ == "__main__":
    main()
