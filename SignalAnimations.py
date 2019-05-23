import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np

class SignalDelayer(object):
    def __init__(self, temporal_frames):
        self.temporal_frames = temporal_frames-1
    def SetData(self, x, y):
        self.x = x
        self.y = y
    def SetCol(self, color):
        self.color = color
    def TemporalDelay(self, frame):
        lenght = len(self.y)
        newidx = int(lenght*(frame+1)/self.temporal_frames)
        return self.x[:newidx], self.y[:newidx]

def init():
    temp = ax.plot([], [], 'r', alpha=1)
    lines.extend([temp[0]])
    return lines

def update(frame, x, y):
    if frame<signals*temporal_frames-1:
        timestep = frame%temporal_frames
        if timestep<temporal_frames-1:
            # lines = [i for i in lines]
            if timestep==0:
                # categorical = np.random.randint(4)
                categorical = list_classes[frame//temporal_frames]
                print("demo: ", frame, frame//temporal_frames)
                idx = np.random.randint(50)
                tempx, tempy = x, DataX[idx:idx+samples].copy()
                if categorical==0:
                    tempy *= np.random.randn(tempx.shape[0])/4
                elif categorical==1:
                    tempy[np.random.randn(tempx.shape[0])>1.6] = 1
                    tempy += np.random.randn(tempx.shape[0])/100
                delayer.SetData(tempx, tempy)
                delayer.SetCol(colors[categorical])
            temp = ax.plot([], [], delayer.color, alpha=1) # Aqui
            tempx, tempy = delayer.TemporalDelay(timestep)
            temp[0].set_data(tempx, tempy)
            if frame==0: lines[0] = temp[0]
            else: lines.extend(temp)
        else:
            for cnt, line in enumerate(lines[::-1]):
                tempx, tempy = line.get_data()
                line.set_data(tempx, tempy+displacement)
                line.set_alpha(np.minimum(1,4/(cnt+1)))
    elif frame<signals*temporal_frames+generation-1:
        cnt_gen = frame-(signals*temporal_frames-1)
        ax.set_xlim(0-np.exp(cnt_gen/10), samples)
        ax.set_ylim(-1, 13.1+np.exp(cnt_gen/20))
    elif frame==signals*temporal_frames+generation-1:
        lines.clear()
        ax.clear()
        n = 100
        cnt_gen = frame-(signals*temporal_frames-2)
        for j in range(len(colors)):
            groups.append(     np.random.multivariate_normal(centroids[j],[[2,1],[3,1]],[50])    )
            temp = ax.scatter(groups[j][:,0], groups[j][:,1], s=50, c=colors[j])
            ax.set_xlim(0-np.exp(cnt_gen/10), samples)
            ax.set_ylim(-4, 13.1+np.exp(cnt_gen/20))
            ax.set_xticks([])
            ax.set_yticks([])
            lines.extend([temp])
            mx, my = groups[j].mean(0)
            X = np.linspace(mx,ends[j][0], int(clustering*0.7))
            Y = np.linspace(my,ends[j][1], int(clustering*0.7))
            X = np.concatenate((X,[ends[j][0]+np.random.rand()/10]*int(clustering*0.3)))
            Y = np.concatenate((Y,[ends[j][1]+np.random.rand()/10]*int(clustering*0.3)))
            paths.append(np.array([X,Y]))
        print(frame, "lines: " , len(lines))
    elif frame<signals*temporal_frames+generation*transition:
        cx1, cx2 = ax.get_xlim()
        cy1, cy2 = ax.get_ylim()
        ax.set_xlim(cx1/1.066,cx2/1.025)
        ax.set_ylim(cy1,cy2/1.024)
        # print("resclaing...", frame)
    elif frame<signals*temporal_frames+generation*transition+clustering-1:
        # print(frame, frame-signals*temporal_frames-generation*transition)
        movestep = frame-signals*temporal_frames-generation*transition
        # print(movestep)
        for cnt, line in enumerate(lines):
            data = line.get_offsets()
            rn = np.random.rand(2)/10
            moves = paths[cnt][:,movestep]+rn - data.mean(0)
            data += moves
            line.set_offsets(data+ np.random.rand(50,2)/4*(int(movestep<150)))
    else:
        print("finished!!!!!!!!!!!!!!")
        lines.clear()
    # print(frame)
    return lines

# Main clusters
list_classes = [3,2,0,2,4,1,3,0,1,3,2,4,5,5,0,3,1,2]
colors = ['r','g','b','k','c','m']
centroids = [(-5,2),(4,2), (2,5), (-2,5), (0,7), (10,-2)]
ends = [(11,0),(18,12), (-25,8), (-15,9), (-20,5), (-23,4)]
# Signal Properties
samples = 500
Fs = 10000
x = np.arange(samples)
t = np.arange(samples*10)
y = np.sin(2 * np.pi * 100 * t / Fs)
DataX = []
# for j in [Data1, Data2, Data3, Data4]:
for i in y: 
    DataX.append(i)
DataX = np.array(DataX)

# Timing frames per stage... VERY IMPORTANT FOR VISUALIZATION
temporal_frames=13
generation = 100
transition=2
clustering = 250
delayer = SignalDelayer(temporal_frames)
displacement = 1. # Vertical movement for signals
lines = []
groups = []
paths = []
# Figure properties
fig, ax = plt.subplots(1)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
signals = len(list_classes)
ax.set_ylim(-1.,13.1)
ax.set_xlim(0,samples)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)



ani = animation.FuncAnimation(fig, update, init_func=init, frames=signals*temporal_frames+generation*transition+clustering, fargs=(x, DataX), blit=True, interval=1, repeat=False)
lines = []
# fig.show()
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='EdgarMedina'), bitrate=1800)
ani.save('simple_animation.mp4', writer=writer)

