# hand_avoid_game_shoot_octagon.py
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math

# --------------------------
# Config
# --------------------------
WIDTH, HEIGHT = 640, 480
PLAYER_RADIUS = 25
PLAYER_Y = HEIGHT - 70
ENEMY_SIZE = 50
ENEMY_BASE_SPEED = 3.0
ITEM_RADIUS = 15
SPLIT_DISTANCE = 100
SPLIT_COUNT = 2
DUR_INVULN = 3.0
DUR_SLOW = 3.0
DUR_SCORE_DOUBLE = 5.0
READY_DURATION = 1.3
GO_DURATION = 0.7
HOMING_TURN_SPEED = 0.12
HOMING_MAX_SPEED = 7.0
HOMING_FALL_SPEED = 0.8
SHOT_COOLDOWN = 0.3

# Colors
COL_WHITE = (255,255,255)
COL_GRAY = (100,100,100)
COL_RED = (0,0,255)
COL_BLUE = (255,0,0)
COL_PURPLE = (128,0,255)
COL_ORANGE = (0,165,255)
COL_GREEN = (0,255,0)
COL_YELLOW = (0,255,255)
COL_CYAN = (255,255,0)
COL_BLACK = (0,0,0)

# --------------------------
# Helpers
# --------------------------
def now(): return time.time()
def clamp(v,a,b): return max(a,min(b,v))
def distance(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])
def rect_circle_collision(rx,ry,rw,rh,cx,cy,cr):
    closest_x = clamp(cx, rx, rx+rw)
    closest_y = clamp(cy, ry, ry+rh)
    dx = cx - closest_x
    dy = cy - closest_y
    return dx*dx + dy*dy <= cr*cr

# --------------------------
# Game State
# --------------------------
effects={'invuln_until':0,'slow_until':0,'score_double_until':0}
enemy_list=[]
item_list=[]
bullets=[]
last_enemy_spawn=0
last_item_spawn=0
next_enemy_interval=random.uniform(0.7,1.4)
next_item_interval=random.uniform(6.0,10.0)
game_over=False
score=0
start_time=now()
player_x=WIDTH//2
intro_state='READY'
intro_start=now()
last_shot_time=0.0

# --------------------------
# Mediapipe & Camera
# --------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6,max_num_hands=1)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# --------------------------
# Spawn functions
# --------------------------
def spawn_enemy():
    etype = random.choices(population=['A','B','C','D'],weights=[0.5,0.2,0.2,0.1])[0]
    x = random.randint(0, WIDTH-ENEMY_SIZE)
    y = -ENEMY_SIZE
    speed = ENEMY_BASE_SPEED
    if etype=='B': speed*=1.8
    if etype=='C': speed*=0.9
    return {'type':etype,'x':float(x),'y':float(y),'size':ENEMY_SIZE,'speed':float(speed),'vx':0.0,'vy':0.0,'dir_angle':None,'split':False}

def spawn_item():
    itype = random.choice(['HEAL','SLOW','STAR','SCORE'])
    x = random.randint(20, WIDTH-20)
    y = -20
    return {'type':itype,'x':float(x),'y':float(y),'vy':2.0}

def reset_game():
    global enemy_list,item_list,bullets,last_enemy_spawn,last_item_spawn
    global next_enemy_interval,next_item_interval,game_over,score,start_time,effects
    global intro_state,intro_start,player_x,last_shot_time
    enemy_list=[]
    item_list=[]
    bullets=[]
    last_enemy_spawn=now()
    last_item_spawn=now()
    next_enemy_interval=random.uniform(0.7,1.4)
    next_item_interval=random.uniform(6.0,10.0)
    game_over=False
    score=0
    start_time=now()
    effects={'invuln_until':0,'slow_until':0,'score_double_until':0}
    intro_state='READY'
    intro_start=now()
    player_x = WIDTH//2
    last_shot_time=0.0

# --------------------------
# Player shooting helpers
# --------------------------
def is_hand_closed(hand_lms):
    wrist = hand_lms.landmark[mp_hands.HandLandmark.WRIST]
    tips = [hand_lms.landmark[i] for i in [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.THUMB_TIP
    ]]
    closed_count=0
    for tip in tips:
        dx, dy = tip.x - wrist.x, tip.y - wrist.y
        if math.hypot(dx,dy)<0.12: closed_count+=1
    return closed_count>=4

def draw_octagon(frame,x,y,size,color):
    pts=[]
    for i in range(8):
        ang = math.radians(45*i)
        px = int(x + size*math.cos(ang))
        py = int(y + size*math.sin(ang))
        pts.append((px,py))
    cv2.fillPoly(frame,[np.array(pts,np.int32)],color)

# --------------------------
# Homing enemy update
# --------------------------
def update_homing_enemy(e,px,py,slow_mult):
    ex,ey = e['x'], e['y']
    dx,dy = px-ex, py-ey
    dist = max(math.hypot(dx,dy),1.0)
    target_angle = math.atan2(dy,dx)
    if e['dir_angle'] is None: e['dir_angle']=target_angle
    diff = (target_angle - e['dir_angle'] + math.pi)%(2*math.pi)-math.pi
    diff = clamp(diff,-HOMING_TURN_SPEED,HOMING_TURN_SPEED)
    e['dir_angle'] += diff
    speed = min(e['speed'],HOMING_MAX_SPEED)*slow_mult
    if dist<120: speed*=0.4
    e['x'] += math.cos(e['dir_angle'])*speed
    e['y'] += math.sin(e['dir_angle'])*speed
    e['y'] += HOMING_FALL_SPEED*slow_mult

# --------------------------
# Main loop
# --------------------------
reset_game()
while True:
    ret,cam_frame = cap.read()
    if not ret: break
    cam_frame = cv2.flip(cam_frame,1)
    game_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    game_frame[:] = 0
    game_frame = cv2.cvtColor(game_frame, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    t = now()

    # Update player
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        wrist = lm.landmark[mp_hands.HandLandmark.WRIST]
        player_x = int(wrist.x*WIDTH)
        if is_hand_closed(lm) and t-last_shot_time>=SHOT_COOLDOWN:
            bullets.append({'x':player_x,'y':PLAYER_Y-PLAYER_RADIUS,'vy':-10.0})
            last_shot_time = t

    # Intro
    if intro_state=='READY' and t-intro_start>=READY_DURATION:
        intro_state='GO'; intro_start=now()
    elif intro_state=='GO' and t-intro_start>=GO_DURATION:
        intro_state='PLAY'; intro_start=now(); last_enemy_spawn=now(); last_item_spawn=now()

    # Gameplay
    if intro_state=='PLAY' and not game_over:
        slow_mult = 0.5 if t<effects['slow_until'] else 1.0

        # Spawn enemies
        if t-last_enemy_spawn>=next_enemy_interval:
            enemy_list.append(spawn_enemy())
            last_enemy_spawn=t
            next_enemy_interval=random.uniform(0.7,1.4)
        # Spawn items
        if t-last_item_spawn>=next_item_interval:
            item_list.append(spawn_item())
            last_item_spawn=t
            next_item_interval=random.uniform(6.0,10.0)

        # Update enemies
        new_enemy_list=[]
        for e in enemy_list:
            et=e['type']
            if et in ('A','B'): e['y'] += e['speed']*slow_mult
            elif et=='C': update_homing_enemy(e,player_x,PLAYER_Y,slow_mult)
            elif et=='D':
                e['y'] += e['speed']*0.8*slow_mult
                if not e.get('split',False):
                    if distance((e['x']+e['size']/2,e['y']+e['size']/2),(player_x,PLAYER_Y))<=SPLIT_DISTANCE:
                        for i in range(SPLIT_COUNT):
                            offset=(i-(SPLIT_COUNT-1)/2)*(e['size']/2)
                            nx = clamp(int(e['x']+offset),0,WIDTH-ENEMY_SIZE)
                            ny = int(e['y'])
                            new_enemy_list.append({'type':'A','x':float(nx),'y':float(ny),'size':ENEMY_SIZE//2,'speed':e['speed']+1.2,'vx':0.0,'vy':0.0,'dir_angle':None,'split':True})
                        e['split']=True
                        continue
            if e['y']<HEIGHT+200: new_enemy_list.append(e)
        enemy_list = new_enemy_list

        # Update bullets
        new_bullets=[]
        for b in bullets:
            b['y'] += b['vy']
            hit_enemy=None
            for e in enemy_list:
                ex,ey,sz=e['x'],e['y'],e.get('size',ENEMY_SIZE)
                if e['type'] in ('A','B','D'):
                    if rect_circle_collision(ex,ey,sz,sz,b['x'],b['y'],5): hit_enemy=e; break
                elif e['type']=='C':
                    cr=sz/2; cx=ex+cr; cy=ey+cr
                    if distance((b['x'],b['y']),(cx,cy))<=cr+5: hit_enemy=e; break
            if hit_enemy:
                try: enemy_list.remove(hit_enemy)
                except ValueError: pass
                continue
            if 0<=b['y']<=HEIGHT: new_bullets.append(b)
        bullets=new_bullets

        # Enemy collision with player
        if t>=effects['invuln_until']:
            for e in enemy_list:
                et=e['type']
                if et in ('A','B','D'):
                    if rect_circle_collision(e['x'],e['y'],e.get('size',ENEMY_SIZE),e.get('size',ENEMY_SIZE),player_x,PLAYER_Y,PLAYER_RADIUS):
                        game_over=True; break
                elif et=='C':
                    cr=e.get('size',ENEMY_SIZE)/2; cx=e['x']+cr; cy=e['y']+cr
                    if distance((cx,cy),(player_x,PLAYER_Y))<=cr+PLAYER_RADIUS: game_over=True; break

        # Score
        score = int((now()-start_time)*(2 if t<effects['score_double_until'] else 1))

    # --------------------------
    # Drawing
    # --------------------------
    # Player
    if now()<effects['invuln_until'] and int(now()*5)%2==0:
        cv2.circle(game_frame,(player_x,PLAYER_Y),PLAYER_RADIUS,COL_WHITE,-1)
    else:
        cv2.circle(game_frame,(player_x,PLAYER_Y),PLAYER_RADIUS,COL_WHITE,-1)

    # Enemies
    for e in enemy_list:
        ex,ey,sz = int(e['x']),int(e['y']),int(e.get('size',ENEMY_SIZE))
        if e['type']=='A': cv2.rectangle(game_frame,(ex,ey),(ex+sz,ey+sz),COL_RED,-1)
        elif e['type']=='B': cv2.rectangle(game_frame,(ex,ey),(ex+sz,ey+sz),COL_BLUE,-1)
        elif e['type']=='C':
            cr=sz//2; cx=ex+cr; cy=ey+cr
            cv2.circle(game_frame,(cx,cy),cr,COL_PURPLE,-1)
        elif e['type']=='D':
            cx=ex+sz//2; cy=ey+sz//2; rr=sz//2
            pts=[]
            for k in range(6):
                ang=math.radians(60*k)
                pts.append((int(cx+rr*math.cos(ang)),int(cy+rr*math.sin(ang))))
            cv2.fillPoly(game_frame,[np.array(pts,np.int32)],COL_ORANGE)

    # Items
    for it in item_list:
        ix,iy=int(it['x']),int(it['y'])
        if it['type']=='HEAL': cv2.circle(game_frame,(ix,iy),ITEM_RADIUS,COL_GREEN,-1)
        elif it['type']=='SLOW': cv2.circle(game_frame,(ix,iy),ITEM_RADIUS,COL_YELLOW,-1)
        elif it['type']=='STAR': cv2.circle(game_frame,(ix,iy),ITEM_RADIUS,COL_WHITE,-1)
        elif it['type']=='SCORE': cv2.circle(game_frame,(ix,iy),ITEM_RADIUS,COL_CYAN,-1)

    # Bullets (small octagon)
    for b in bullets:
        draw_octagon(game_frame,b['x'],b['y'],8,COL_WHITE)

    # HUD
    cv2.putText(game_frame,f"Score: {score}",(10,36),cv2.FONT_HERSHEY_SIMPLEX,1.0,COL_WHITE,2)
    ef_texts=[]
    if now()<effects['invuln_until']: ef_texts.append(f"Invuln:{int(effects['invuln_until']-now())+1}s")
    if now()<effects['slow_until']: ef_texts.append(f"Slow:{int(effects['slow_until']-now())+1}s")
    if now()<effects['score_double_until']: ef_texts.append(f"Score x2:{int(effects['score_double_until']-now())+1}s")
    if ef_texts: cv2.putText(game_frame," | ".join(ef_texts),(10,66),cv2.FONT_HERSHEY_SIMPLEX,0.7,COL_GRAY,2)

    # Ready/Go/GameOver
    if intro_state=='READY':
        cv2.putText(game_frame,"READY",(int(WIDTH*0.35),int(HEIGHT*0.35)),cv2.FONT_HERSHEY_SIMPLEX,2.5,COL_YELLOW,6)
    elif intro_state=='GO':
        cv2.putText(game_frame,"GO!",(int(WIDTH*0.42),int(HEIGHT*0.35)),cv2.FONT_HERSHEY_SIMPLEX,2.5,COL_GREEN,6)
    elif intro_state=='PLAY' and game_over:
        cv2.putText(game_frame,"GAME OVER",(int(WIDTH*0.25),int(HEIGHT*0.35)),cv2.FONT_HERSHEY_SIMPLEX,2.0,COL_YELLOW,5)
        cv2.putText(game_frame,"Press SPACE to Restart",(int(WIDTH*0.22),int(HEIGHT*0.45)),cv2.FONT_HERSHEY_SIMPLEX,0.9,(200,200,200),2)

    cv2.putText(game_frame,"ESC or Q: Quit",(10,HEIGHT-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,COL_GRAY,2)

    cv2.imshow("Hand Avoid Game - Shoot Octagon",game_frame)

    key=cv2.waitKey(1) & 0xFF
    if key==27 or key==ord('q'): break
    if key==ord(' ') and (game_over or intro_state!='PLAY'): reset_game()

cap.release()
cv2.destroyAllWindows()