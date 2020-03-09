import os
import numpy as np

def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

xml_head = \
"""
<scene version="0.5.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="ldrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.02"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
            <scale value="0.7"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="10"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]

def mitsuba(pcl, path, clr=None):
    xml_segments = [xml_head]

    # pcl = standardize_bbox(pcl, 2048)
    pcl = pcl[:,[2,0,1]]
    pcl[:,0] *= -1
    h = np.min(pcl[:,2])

    for i in range(pcl.shape[0]):
        if clr == None:
            color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5)
        else:
            color = clr
        if h < -0.25:
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2]-h-0.6875, *color))
        else:
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    with open(path, 'w') as f:
        f.write(xml_content)

if __name__ == '__main__':   
    item = 0
    split = 'train'
    dataset_name = 'shapenetcorev2'
    root = os.getcwd()
    save_root = os.path.join("image", dataset_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    from dataset import Dataset
    d = Dataset(root=root, dataset_name=dataset_name, 
                        num_points=2048, split=split, random_rotation=False, load_name=True)
    print("datasize:", d.__len__())

    pts, lb, n = d[item]
    print(pts.size(), pts.type(), lb.size(), lb.type(), n) 
    path = os.path.join(save_root, dataset_name + '_' + split + str(item) + '_' + str(n) + '.xml')
    mitsuba(pts.numpy(), path)
