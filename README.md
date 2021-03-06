# pytorch-fsim
Differentiable implementation of the Feature Similarity Index Measure in Pytorch with CUDA support

# Installation
* Clone the repository
* pip3 install -r requirements.txt

# Basic usage

## Computing score
```
# Path to reference image
img1_path ='./misc/mandril_color.tif'
# Is it black and white?
bw = False
# Size of the batch for training
batch_size = 1

# Read reference and distorted images
img1 = Image.open(img1_path).convert('RGB')
img1 = pt.from_numpy(np.asarray(img1))
img1 = img1.permute(2,0,1)
img1 = img1.unsqueeze(0).type(pt.FloatTensor)
img2 = pt.clamp(pt.rand(img1.size())*255.0,0,255.0)

# Create fake batch (for testing)
img1b = pt.cat(batch_size*[img1],0)
img2b = pt.cat(batch_size*[img2],0)

if pt.cuda.is_available():
    img1b = img1b.cuda()
    img2b = img2b.cuda()

# Create FSIM loss
FSIM_loss = FSIMc()
loss = FSIM_loss(img1b,img2b)    
print(loss)
```

## Optimizing 
```
# Path to reference image
img1_path ='./misc/mandril_color.tif'
# Is it black and white?
bw = False
# Size of the batch for training
batch_size = 1

# Read reference and distorted images
img1 = Image.open(img1_path).convert('RGB')
img1 = pt.from_numpy(np.asarray(img1))
img1 = img1.permute(2,0,1)
img1 = img1.unsqueeze(0).type(pt.FloatTensor)
img2 = pt.clamp(pt.rand(img1.size())*255.0,0,255.0)

# Create fake batch (for testing)
img1b = pt.cat(batch_size*[img1],0)
img2b = pt.cat(batch_size*[img2],0)
# Convert images to variables to support gradients
img1b = Variable( img1b, requires_grad = False)
img2b = Variable( img2b, requires_grad = True)

if pt.cuda.is_available():
    img1b = img1b.cuda()
    img2b = img2b.cuda()

# Create FSIM loss
FSIM_loss = FSIMc()

# Tie optimizer to the distorted batch
optimizer = optim.Adam([img2b], lr=0.1)

# Check if the gradient propagates
for ii in range(0,1000):
    optimizer.zero_grad()

    loss = -FSIM_loss(img1b,img2b)    
    print(loss)
    loss = pt.sum(loss)
    loss.backward()
    optimizer.step()
```

# References
The code is the direct implementation of the MATLAB version provided by:

https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/FSIM.htm
