import cv2
import torch
from model import Generator
from torchvision import transforms as T


cap = cv2.VideoCapture(0)
model = Generator().to("cuda")
model.load_state_dict(torch.load("models/pix2pix_20.pt")['generator'])


transforms = T.Compose([
    T.ToTensor(),
    T.Resize((256, 256), antialias=True),
])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t_frame = transforms(frame_rgb)
    t_frame = t_frame.unsqueeze(0).to("cuda")
    
    with torch.no_grad():
        gen_output = model(t_frame).permute(0, 2, 3, 1).cpu().numpy()[0]

    gen_output = (gen_output * 255).astype('uint8')
    gen_output = cv2.cvtColor(gen_output, cv2.COLOR_RGB2BGR)  

    gen_output_resized = cv2.resize(gen_output, (frame.shape[1], frame.shape[0]))
    
    cv2.imshow(f'Original Frame {frame.shape}', frame)
    cv2.imshow('Generated Output', gen_output_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
