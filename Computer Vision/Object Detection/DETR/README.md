


3. 
3.2 DERT architecture 

Three main compoent: 
- CNN backbone : x = H0 * W0 * 3 --> out : H0//32 * W0//32 * 2048 
- Encoder-Decoder Transformer 
Encoder
 Sử dụng Conv 1x1 để giảm channel từ 2048 -> d(nhỏ hơn) 
 encoder kì vọng đầu vào là 1 sequence nên chúng ta thu gọn thành 1 chiều -> d*HW. Mỗi encoder thì đều quy chuẩn là có multihead và FFN. 
 Kể từ khi transformer permuation-invarian. chúng tôi bổ sung position encoding
 Decoder tuân theo cái cấu trúc chuẩn của transformer thôi 
 sử dụng mutilhead và encoder-decoder attention mechanisms. Sự khác biệt là chúng tôi decode N object song song ở mối lớp  	
- FFN 




4. So sánh với Faster RCNN trên tập COCO. 
Chi tiết về DETR: 
- Loss: AdamW (lr = 10e-4) 
- Xavier Init weight 
- Backbone: ImageNet-pretrained ResNet 50 vs 101 
4.1 
Transformer using Adam or Adagrad optimize với training schedules dài và dropout. còn Fater RCNN train với SGD 
