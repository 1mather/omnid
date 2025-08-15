import asyncio
import numpy as np
import websockets
import cv2
import base64
import logging
import traceback
import sys
import msgpack_numpy
import msgpack

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

async def send_test_request():
    uri = "ws://127.0.0.1:8000"  # 使用服务器端口
    
    logging.info(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            logging.info("Connected to server")
            
            # 创建测试图像
            # 创建测试图像，保持int8类型
            test_image = np.zeros(( 3, 480, 480), dtype=np.float32)
            #test_image = np.zeros((480, 480, 3), dtype=np.uint8) 
            
            # 创建状态向量
            state = np.array([0.97, 0.68, 0.77, 0.59, 0.27, 0.70, 0.63], dtype=np.float32)
            
            # 创建观察数据，使用正确的键名
            observation = {
                "observation.image_0": test_image,  # 直接使用NumPy数组，不进行编码
                "observation.state": state
            }
            
            # 使用msgpack_numpy序列化数据
            packed_data = msgpack_numpy.packb(observation, use_bin_type=True)
            
            # 发送请求
            logging.info("Sending msgpack request...")
            await websocket.send(packed_data)
            
            # 接收响应
            logging.info("Waiting for response...")
            response = await websocket.recv()
            
            # 解析响应 (假设响应也是msgpack格式)
            try:
                unpacked_response = msgpack_numpy.unpackb(response, raw=False)
                logging.info(f"Received msgpack response: {unpacked_response}")
                
                if "action" in unpacked_response:
                    action = unpacked_response["action"]
                    logging.info(f"Action values: {action}")
                    return action
                else:
                    logging.warning(f"Unexpected response format: {unpacked_response}")
                    return None
            except Exception as e:
                # 如果响应不是msgpack格式，尝试作为字符串处理
                logging.error(f"Failed to unpack response: {e}")
                logging.error(f"Raw response: {response}")
                return None
            
    except Exception as e:
        logging.error(f"Error: {e}")
        logging.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    try:
        result = asyncio.run(send_test_request())
        if result is not None:
            logging.info(f"Test completed successfully with result: {result}")
        else:
            logging.error("Test failed")
    except KeyboardInterrupt:
        logging.info("Test interrupted by user")
    except Exception as e:
        logging.error(f"Unhandled error: {e}")
        logging.error(traceback.format_exc())