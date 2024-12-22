"""
Script per creazione dell'ambiente.
Obiettivo del robot: Alzare l'oggetto ad un'altezza specifica

---------------------------------------------------------------------------------------------------------------
Giunti controllabili del robot:
nome_robot: Tipo_giunto (indice) ---> Gli angoli dei giunti rotoidali sono in radianti
panda_joint1: Revolute (0)
panda_joint2: Revolute (1)
panda_joint3: Revolute (2)
panda_joint4: Revolute (3)
panda_joint5: Revolute (4)
panda_joint6: Revolute (5)
panda_joint7: Revolute (6)
panda_finger_joint1: Prismatic (9)
panda_finger_joint2: Prismatic (10)

--------------------------------------------------------------------------------------------------------------
Azioni possibili:
[0, num_azioni_x x num_azioni_y x num_angolazioni] :
ad ogni "marco-pixel" dell'osservazione è associato un leggero spostamento del robot
rispetto alla coordinata del pixel centrale.
---------------------------------------------------------------------------------------------------------------
Osservazione:
Immagine binaria di dimensioni n x n pixel. Per ora n = 37

---------------------------------------------------------------------------------------------------------------
Reward:
+1 oggetto sollevato
-2 oggetto non sollevato

possibile modifica: reward in funzione della differenza con la posizione desiderata
---------------------------------------------------------------------------------------------------------------

TODO:
 - provare a inserire elementi di forme diverse
 - Aumentare numero di angoli
 - Considerare la postura dell'oggetto all'istante finale

"""

"""
    LIBRERIE
"""
import random
import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import cv2
from gymnasium import spaces
import time


class RobotEnv(gym.Env):
    """
    Classe che rappresenta l'ambiente per l'addestramento del robot utilizzando PyBullet.
    L'ambiente supporta sia la modalità con GUI che senza GUI, ed è configurabile per l'uso in tempo reale o meno.
    """

    def __init__(self, is_for_training=False):
        """
        Inizializza l'ambiente del robot.

        Args:
            is_for_training (bool): Se True, utilizza l'interfaccia grafica di PyBullet e
                                    la simulazione viene eseguita in tempo reale.
        """
        super(RobotEnv, self).__init__()

        """
        ---------------------------------------------------------------------------------------------------------------
        VARIABILI
        ---------------------------------------------------------------------------------------------------------------
        """
        # ------------------------------------------------------------------------------------------------------------

        # Variabili che regolano i diversi aspetti dell'ambiente per l'allenamento e per la simulazione
        self.use_gui = not is_for_training
        self.realtime = not is_for_training

        # Variabili per l'ambiente
        self.num_step_max_per_episode = 1  # un passo per episodio --> provo ad afferare e ripeto il ciclo
        self.num_step = 0  # numero di step fatti nell'episodio
        self.num_simulation_steps = 2000  # numero massimo di step di simulazione per il movimento del robot
        self.tolleranza = 0.005  # Valore di tolleranza per il moto del robot
        self.tolleranza_reward = 0.05

        # --------------------------------------------OGGETTI-------------------------------------------------------

        # Variabili per gli oggetti
        self.obj_size = np.array([0.02, 0.05, 0.02])  # dimensioni
        self.obj_x_min = -0.15  # coordinate massime e minime
        self.obj_x_max = 0.15
        self.obj_y_min = 0.40
        self.obj_y_max = 0.70
        # Commentato perché, per ora, l'e.e. e gli oggetti possono assumere le stesse orientazioni
        # self.obj_yaw = [0, -np.pi / 4, -np.pi / 2, np.pi / 4]  # Postura dell'oggetto (r,p,y)
        self.obj_pitch = 0
        self.obj_roll = 0

        # ------------------------------------------CAMERA------------------------------------------------------------

        # Coordinate puntate dalle camere (nel centro delle possibili posizioni degli oggetti)
        camera_point_to_x = np.mean([self.obj_x_min, self.obj_x_max])
        camera_point_to_y = np.mean([self.obj_y_min, self.obj_y_max])
        self.camera_point_to = [camera_point_to_x, camera_point_to_y, 0]

        # Dimensioni immagine catturata
        self.width_img = 401
        self.height_img = 401

        # Campo visivo (FOV) della telecamera per le osservazioni in radianti
        field_of_view = 60
        fov_rad = np.radians(field_of_view)

        # Distanza della camera per le osservazioni dalle coordinate puntate
        self.camera_distance = 0.4

        # --------------------------------------------ROBOT-----------------------------------------------------------

        # Robot

        # Posizioni iniziali giunti robot
        self.joint_positions_start = [0.0, 0.0, 0.0, -np.pi / 3, 0.0, np.pi / 1.75, np.pi / 4, 0.0, 0.0, 0.03, 0.03,
                                      0.0]
        self.joint_angles_des = None
        self.joint_velocities = np.array([0.1, 0.1, 0.1, 0.1, 0.20, 0.25, 0.55, 0.0, 0.0, 0.0, 0.0]) * 2

        # self.angles = [0, -np.pi / 4, -np.pi / 2, np.pi / 4]
        self.angles = [0, -np.pi / 6, -np.pi / 3, -np.pi / 2,  2 * np.pi / 6, np.pi / 6]
        self.num_angolazioni = len(self.angles)
        self.obj_yaw = self.angles              # Postura dell'oggetto (r,p,y)

        # --------------------------------------------AZIONI E OSSERVAZIONI--------------------------------------------

        # Grandezze per l'osservazione e l'obiettivo dell'agente
        ##############
        self.place_pos = [0.1, 0, 0.3]
        ############
        self.altezza_cubo_desiderata = 0.50
        self.obs_width = 37
        self.obs_height = 37
        self.num_azioni_x = 7
        self.num_azioni_y = 7
        self.posizione_centrale_obs = np.array([0, 0])
        self.delta_l = 0                                    # Unità unitaria di quanto si deve spostare il robot

        # Calcola la metà dell'ampiezza della visuale nel mondo reale
        self.semi_ampiezza = self.camera_distance * np.tan(fov_rad / 2)

        # Coordinate minime e massime per (x,y), puntate dalla camera delle osservazioni
        self.world_x_min = self.camera_point_to[0] - self.semi_ampiezza
        self.world_x_max = self.camera_point_to[0] + self.semi_ampiezza
        self.world_y_min = self.camera_point_to[1] - self.semi_ampiezza
        self.world_y_max = self.camera_point_to[1] + self.semi_ampiezza

        # ---------------------------------------------------CREAZIONE AMBIENTE--------------------------------------

        """
        Connessione a PyBullet
        """
        try:
            if self.use_gui:
                p.connect(p.GUI)
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.6,
                    cameraYaw=120,
                    cameraPitch=-50,
                    cameraTargetPosition=self.camera_point_to
                )
            else:
                p.connect(p.DIRECT)
        except Exception as e:
            print(f"Errore durante la connessione a PyBullet: {e}")
            raise

        """
        Camera dalla quale vengono ottenute le osservazioni
        """
        # Imposta la telecamera per la visualizzazione
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_point_to,
            distance=self.camera_distance,  # Distanza della telecamera
            yaw=0,
            pitch=-90,
            roll=0,
            upAxisIndex=2  # L'asse Z è considerato l'asse "su"
        )

        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=field_of_view,  # Campo visivo della telecamera
            aspect=float(self.width_img) / self.height_img,  # Rapporto d'aspetto
            nearVal=0.1,  # Distanza minima di clipping
            farVal=5.0  # Distanza massima di clipping
        )

        """
        Caricamento degli elementi presenti nell'ambiente:
            - piano
            - robot
            - oggetti
        """
        # Imposta il percorso per i file URDF
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Carica il piano
        try:
            self.planeId = p.loadURDF("plane.urdf")
        except Exception as e:
            print(f"Errore durante il caricamento dell'URDF del piano: {e}")
            raise

        # Definizione degli oggetti disponibili con un dizionario che mappa gli indici ai metodi di creazione
        oggetti_disponibili = {
            0: self.crea_oggetto_l,
            1: self.crea_parallelepipedo,
            2: self.crea_oggetto_z
        }

        # Genera la posizione e l'orientazione dell'oggetto
        pos_obj, orientazione_obj = self.genera_nuova_posa()

        # Generazione casuale o selezione controllata del tipo di oggetto
        # tipo_obj = random.randint(0, len(oggetti_disponibili) - 1)  # Seleziona casualmente uno tra gli oggetti
        tipo_obj = 0

        # Controllo se l'oggetto scelto è valido
        try:
            self.objId = oggetti_disponibili[tipo_obj](pos_obj, orientazione_obj)
        except KeyError:
            print(f"Errore: il tipo di oggetto selezionato ({tipo_obj}) non è valido. Selezionare un tipo valido.")
            raise

        # Carica il robot
        orientation_quat = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        try:
            self.robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, baseOrientation=orientation_quat)
        except Exception as e:
            print(f"Errore durante il caricamento dell'URDF del robot: {e}")
            raise

        # Salva il numero di giunti del robot
        self.num_tot_joints = p.getNumJoints(self.robotId)  # Ci sono dei giunti che non sono controllabili

        # Resetta la posa del robot
        self.reset_robot()

        """
        Impostazioni per la gravità e la simulazione dell'ambiente
        """
        # Imposta la gravità e il timestep della simulazione
        p.setGravity(0, 0, -9.81)
        self.timestep = (1. / 75. if is_for_training else 1. / 240.)  # Intervallo di tempo tra passi della simulazione
        p.setTimeStep(self.timestep)

        # Abilita o disabilita la simulazione in tempo reale
        p.setRealTimeSimulation(1 if self.realtime else 0)

        """
        Spazio delle osservazioni e delle azioni
        """
        # Definizione dello spazio delle azioni e dello spazio delle osservazioni
        self.action_space = spaces.Discrete(self.num_azioni_x * self.num_azioni_y * self.num_angolazioni)

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.obs_width, self.obs_height, 1),
                                            dtype=np.uint8)  # Immagine 37*37 binaria

    def reset(self, seed=None, options=None):
        """
        Reimposta l'ambiente all'inizio di un nuovo episodio.

        Args:
            seed (int, opzionale): Seed per la generazione casuale della posa del cubo
                                    In implementazioni future potrà influenzare anche il numero
                                    e la dimensione degli oggetti.
                                    Di default None.
            options (dict, opzionale): Opzioni per il reset. Di default None.

        Returns:
            osservazione (np.array): Osservazione dell'ambiente.
            dict: Informazioni aggiuntive (vuoto per ora ma può tornare comodo).
        """

        if seed is not None:
            np.random.seed(seed)  # Imposta il seed per la riproducibilità

        # Resetta il numero di passi
        self.num_step = 0

        # Resetta la posa del cubo
        pos_obj, orientazione_obj = self.genera_nuova_posa()
        p.resetBasePositionAndOrientation(self.objId, pos_obj, orientazione_obj)

        # Resetta la posa del robot
        self.reset_robot()

        return self.get_observation(), {}

    def step(self, action):
        """
        Esegue un passo della simulazione in base all'azione fornita.

        Args:
            action (int): L'azione da eseguire: un intero che codifica un pixel e un'orientazione

        Returns:
            tuple: Osservazione, ricompensa, flag di completamento, flag di troncamento, informazioni aggiuntive.
        """
        done = False

        # Incrementa il numero di step
        self.num_step += 1

        posizione_target, angolazione = self.action_to_position_and_observation(action)
        # Esegue il movimento desiderato del robot
        self.move_the_robot(posizione_target, angolazione)

        # Ottiene l'osservazione successiva
        observation = self.get_observation()

        # REWARD
        reward = self.calculate_reward()

        # Utile se si deciderà di mettere più step per episodio
        if self.num_step >= self.num_step_max_per_episode:
            done = True

        return observation, reward, done, False, {}

    def close(self):
        """
        Chiude la connessione a PyBullet.
        """
        p.disconnect()

    def calculate_reward(self):
        """
        Calcola la reward.

        Migliorabile fornendo una reward più granulare, specie se si vuole imporre una posa all'oggetto

        Returns:
            reward (double): positiva se l'oggetto è stato sollevato, negativo viceversa
        """
        # REWARD
        # Ricompensa basata sul sollevamento del cubo
        obj_pos, _ = p.getBasePositionAndOrientation(self.objId)
        obj_pos = np.array(obj_pos)  # Convert tuple or list to NumPy array
        place_pos = np.array(self.place_pos)  # Convert tuple or list to NumPy array
        distanza = np.linalg.norm(obj_pos - place_pos)

        if distanza < self.tolleranza_reward:
            reward = 1
        else:
            # reward = -np.linalg.norm(obj_pos[2] - self.altezza_cubo_desiderata)
            reward = -2

        return reward

    def action_to_position_and_observation(self, action):
        """
        Traduce l'azione in una posizione e un'orientazione
        Es.4 orientazioni possibili
        action = 0,1,2,3,4,5,6,7,8,9,...
            [0,1,2,3] -> angolazioni α, β, γ, δ	 associate al macro-pixel 0-esimo dell'osservazione
            [4,5,6,7] -> angolazioni α, β, γ, δ	 associate al macro-pixel 1-esimo dell'osservazione
            ...

        Es. 3 macro-pixel per dimensione nell'immagine, si associa a ogni macro-pixel un identificativo:
            0 1 2
            3 4 5
            6 7 8
        grazie a questi identificativi, si riesce a correggere la posizione del gripper

        Args:
            action (int): L'azione da eseguire: un intero fornito dal modello allenato

        Returns:
            posizione_target: Posizione che il robot deve raggiungere per una presa efficiente
            orientazione_target: Orientazione che il robot deve raggiungere per una presa efficiente
        """

        # Decodifica angolo
        angolazione = self.angles[action % self.num_angolazioni]

        # Decodifica pixel
        # action = 0,1,2,3,4,5,6,7,8,9,... --> 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, ...
        action = action // self.num_angolazioni

        # Mappa l'identificativo del pixel con le coordinate pixel dell'osservazione
        pixel_puntato = [action // self.num_azioni_x, action % self.num_azioni_x]

        # Calcolo delle coordinate mondo da fornire al robot per eseguire inversione cinematica
        # Occhio a non confonderti!
        # --------> x               ∧ y
        # |                         |
        # |                         |
        # |                         |
        # V y                       --------> x
        # Coordinate pixel          coordinate mondo
        posizione_target_x = self.posizione_centrale_obs[0] + (pixel_puntato[0] - (self.num_azioni_x // 2)) * self.delta_l
        posizione_target_y = self.posizione_centrale_obs[1] + (self.num_azioni_y // 2 - pixel_puntato[1]) * self.delta_l
        posizione_target = np.array([posizione_target_x, posizione_target_y])

        return posizione_target, angolazione

    def get_observation(self):
        """
        Ottiene l'osservazione corrente dell'ambiente sotto forma di immagine e calcola le coordinate
        del pixel centrale
        Returns:
            osservazione (np.array): Immagine binaria rappresentante ostacolo (nero) o assenza d'ostacolo (bianco).
        """

        # Inizializza l'osservazione
        osservazione = None

        # Cattura l'immagine
        rgb_image, gray_image = self.cattura_immagine()

        # Rileva i contorni
        edge, contours = self.rileva_contorni(gray_image)

        # In funzione dei contorni, calcola delta_l e la posizione del pixel centrale in coordinate mondo
        if contours:
            for contour in contours:
                # Trova il pixel in alto a sinistra e dimensioni della bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Pixel centrale
                center_x, center_y = x + w // 2, y + h // 2

                # L'obiettivo è ottenere un'osservazione quadrata:
                # trova la dimensione massima della bounding box che rappresenterà il lato del quadrato
                max_length = max(w, h)  # Lato del quadrato [pixel]

                # Calcola le coordinate del mondo reale dal pixel centrale della bounding box
                self.posizione_centrale_obs = self.pixel_to_world(center_x, center_y)

                # Poiché "w" e "h" non costanti, il è necessario calcolare ogni volta il passo dello spostamento
                # che avviene tra un'azione e la adiacente (per ora delta_l è uguale per x e y)
                self.delta_l = (2 * self.semi_ampiezza * max_length / self.width_img) / self.num_azioni_x  # [world]

                osservazione = self.ritaglia_e_ridimensiona(gray_image, max_length, center_x, center_y)

                # Mostra le immagini
                # cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # cv2.rectangle(rgb_image, (center_x-1, center_y-1), (center_x+1, center_y+1), (0, 0, 255), 1)
                # cv2.imshow("rgb", rgb_image)
                # cv2.imshow("gray", gray_image)
                # cv2.imshow("Osservazione", osservazione)
                # cv2.waitKey()

        else:
            # Nel caso in cui non c'è alcun oggetto trovato, realizza un'immagine interamente bianca
            osservazione = np.full((self.obs_width, self.obs_height, 1), 255, dtype=np.uint8)

        return osservazione

    def cattura_immagine(self):
        """
        Cattura l'immagine RGB e la elabora
        N.B.: openCV lavora con BGR, quindi ci sono delle conversioni da fare (probabilmente da ricontrollare)

        Returns:
            rgb_image (np.array): immagine RGB catturata dalla camera
            gray_image (np.array): immagine in scala di grigi catturata dalla camera
        """
        _, _, rgb_image, _, _ = p.getCameraImage(width=self.width_img, height=self.height_img,
                                                 viewMatrix=self.view_matrix,
                                                 projectionMatrix=self.proj_matrix)

        # Elaborazione dell'immagine con OpenCV per trovare i contorni
        rgb_image = np.array(rgb_image, dtype=np.uint8)[:, :, :3]  # nel caso, elimina il canale alfa
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        return rgb_image, gray_image

    def rileva_contorni(self, gray_image):
        """
        Rileva i contorni dell'oggetti presente nell'immagine.
        Per ora, la bounding box è rettangolare e non inclinata

        Returns:
            edges (np.array): Immagine con i bordi rilevati, che rappresenta i contorni dell'oggetto
            contours (list): Lista dei contorni rilevati. Ogni contorno è una sequenza di punti
                             che definiscono il perimetro dell'oggetto individuato nell'immagine.
        """
        # Immagine con i bordi dell'oggetto trovato
        edges = cv2.Canny(cv2.GaussianBlur(gray_image, (5, 5), 0), 150, 200)

        # Trova il contorno e il punto medio del bounding box
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return edges, contours

    def ritaglia_e_ridimensiona(self, gray_image, max_length, center_x, center_y):
        """
        Realizza l'osservazione finale del robot.
        Ritaglia l'immagine in scala di grigi, la ridimensiona e la filtra per ottenere un'immagine binaria

        Returns:
            osservazione (np.array): osservazione dell'ambiente
        """

        half_size = max_length // 2  # Semi lunghezza del lato del quadrato [pixel]

        # Calcola i limiti del ritaglio, evitando di provare a tagliare oltre i bordi
        start_x = max(0, center_x - half_size)
        end_x = min(self.width_img, center_x + half_size)
        start_y = max(0, center_y - half_size)
        end_y = min(self.height_img, center_y + half_size)

        # Ritaglia l'immagine
        cropped_image = gray_image[start_y:end_y, start_x:end_x]

        # Ridimensiona l'immagine ritagliata a (per ora) 37x37 pixel
        if cropped_image.size > 0:  # Evita che l'immagine sia vuota
            resized_object = cv2.resize(cropped_image, (self.num_azioni_x, self.num_azioni_y),
                                        interpolation=cv2.INTER_LINEAR)
            # Soglia binaria per creare un'immagine in bianco e nero
            _, osservazione = cv2.threshold(resized_object, 50, 255, cv2.THRESH_BINARY)
            osservazione = cv2.resize(osservazione, (self.obs_width, self.obs_height),
                                      interpolation=cv2.INTER_NEAREST)
            osservazione = osservazione[:, :, np.newaxis]  # Aggiungi il canale (37, 37, 1)
        else:
            # Immagine bianca se il ritaglio è vuoto
            osservazione = np.full((self.obs_width, self.obs_height, 1), 255, dtype=np.uint8)

        return osservazione

    def pixel_to_world(self, pixel_x, pixel_y):
        """
        Converte le coordinate di un pixel dell'immagine catturata dalla telecamera nelle coordinate del mondo reale.

        Args:
            pixel_x (int): Coordinata x del pixel nell'immagine (origine in alto a sinistra).
            pixel_y (int): Coordinata y del pixel nell'immagine.

        Returns:
            [world_x, world_y] (list): Coordinate (x, y) nel mondo reale corrispondenti al pixel fornito.
        """
        # Mappa le coordinate pixel in coordinate mondo
        world_x = np.interp(pixel_x, [0, self.width_img - 1], [self.world_x_min, self.world_x_max])
        world_y = np.interp(pixel_y, [0, self.height_img - 1], [self.world_y_max, self.world_y_min])

        return [world_x, world_y]

    def genera_nuova_posa(self):
        """
         Genera la nuova posizione e orientazione dell'oggetto

         Returns:
             pos_obj (list): nuova posizione dell'oggetto
             orientazione_obj (double): nuova orientazione
         """
        # Generazione nuova posizione randomica del cubo e orientazione
        pos_obj = [np.random.uniform(low=self.obj_x_min, high=self.obj_x_max),
                   np.random.uniform(low=self.obj_y_min, high=self.obj_y_max),
                   self.obj_size[2]]

        yaw = random.choice(self.obj_yaw)
        orientazione_obj = p.getQuaternionFromEuler([self.obj_roll,
                                                     self.obj_pitch,
                                                     yaw])
        # print("obj_pos : ",pos_obj)
        return pos_obj, orientazione_obj

    def reset_robot(self):
        """
         Resetta la posizione del robot.
        """
        # Usa resetJointState per impostare direttamente la posizione di ogni giunto
        for joint_index in range(self.num_tot_joints):
            target_position = self.joint_positions_start[joint_index]
            p.resetJointState(self.robotId, joint_index, target_position)

        # Salva le nuove posizioni dei giunti
        for joint_index in range(self.num_tot_joints):
            p.setJointMotorControl2(self.robotId, jointIndex=joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.joint_positions_start[joint_index],
                                    force=500)

        p.stepSimulation()

    def normalize_angle(self, angle):
        """
        Normalizza un angolo nell'intervallo [-pi, pi].
        Args:
            angle (double): angolo da normalizzare
        Returns:
            angolo normalizzato (double): Angolo compreso fra [-pi, pi].
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    def simulate_movement(self):
        """
        Simula l'ultimo movimento imposto al robot.
        """

        # step: i-esima iterazione del loop
        step = 0
        normalized_target_angles = [self.normalize_angle(angle) for angle in self.joint_angles_des[:7]]

        while step < self.num_simulation_steps:  # Cicla fino a che non viene raggiunto il numero massimo di cicli
            step += 1  # tiene conto numero di step
            p.stepSimulation()  # esegue uno step di simulazione

            # Probabilmente da togliere, lo faccio per alleggerire il carico computazionale
            if step % 10:  # Ogni tot step, controlla se il punto è stato raggiunto
                # Controlla gli angoli attuali dei giunti
                current_joint_angles = [p.getJointState(self.robotId, i)[0] for i in range(7)]

                # Normalizza gli angoli target e attuali nell'intervallo [-pi, pi]
                normalized_current_angles = [self.normalize_angle(angle) for angle in current_joint_angles]

                # Calcola la differenza angolare tenendo conto della periodicità
                angle_diff = np.abs(np.array(normalized_current_angles) - np.array(normalized_target_angles))

                # Se tutti gli angoli sono entro la tolleranza, termina il ciclo
                if np.all(angle_diff < self.tolleranza):
                    break

    def calculateIK(self, posizione_target, angolazione):
        """
        Normalizza un angolo nell'intervallo [-pi, pi].

        Args:
            posizione_target (np.array): posizione desiderata
        Returns:
            angolazione (double): Angolo compreso fra [-pi, pi] per l'end effector.
        """
        # Calcola, con IK, gli angoli desiderati dei giunti rispetto alla posizione desiderata
        self.joint_angles_des = p.calculateInverseKinematics(self.robotId, 11, posizione_target,
                                                             p.getQuaternionFromEuler([0, np.pi, angolazione]),
                                                             maxNumIterations=50, residualThreshold=1e-3)
        # print("Calculate obj_pos : ",posizione_target)
        # Per ogni giunto, aggiorna il controllo da farci. In questo caso, controllo di posizione
        for i in range(len(self.joint_angles_des)):
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL,
                                    self.joint_angles_des[i], maxVelocity=self.joint_velocities[i])
        # print('joint angles : ',self.joint_angles_des)

    def openGripper(self, apertura_gripper):
        """
        Apre il gripper

        Args:
            apertura_gripper (double): di quanto si deve aprire il gripper
        """
        # Chiude la pinza (giunti 9 e 10 controllano le dita del panda)
        p.setJointMotorControl2(self.robotId, 9, p.POSITION_CONTROL, apertura_gripper)  # Valore da regolare
        p.setJointMotorControl2(self.robotId, 10, p.POSITION_CONTROL, apertura_gripper)

        for _ in range(10):  # 10 iterazioni bastano
            p.stepSimulation()

    def closeGripper(self, max_force=1.0):
        """
            Normalizza un angolo nell'intervallo [-pi, pi].

            Args:
                max_force (double): forza che deve sentire le dita del gripper per terminare la chiusura
        """
        # Chiude la pinza (giunti 9 e 10 controllano le dita del panda)
        p.setJointMotorControl2(self.robotId, 9, p.POSITION_CONTROL, 0.01)  # Valore da regolare
        p.setJointMotorControl2(self.robotId, 10, p.POSITION_CONTROL, 0.01)
        step = 0

        # Esegue al più 100 passi per la simulazione
        while step < 100:
            step += 1
            forces = []
            for joint_index in [9, 10]:
                # Lettura forze sui giunti
                joint_state = p.getJointState(self.robotId, joint_index)
                joint_force = joint_state[3]  # Il quarto elemento è la forza
                forces.append(joint_force)

            # Se una delle forze supera il limite, fermiamo la chiusura
            if any(abs(force) > max_force for force in forces):
                break

            p.stepSimulation()  # Continua a simulare

    def move_the_robot(self, posizione_target, angolazione):
        """
            Fa eseguire al robot il movimento desiderato.
            Per ora si posiziona sopra all'oggetto,
            1) si abbassa
            2) esegue la presa
            3) alza l'oggetto

            Args:
                posizione_target (np.array): posizione da far raggiungere al robot
                angolazione (double): angolo dell'end effector
        """

        # Si posiziona sopra l'oggetto
        new_target = np.array([posizione_target[0],
                               posizione_target[1],
                               self.obj_size[2] * 4])
        self.calculateIK(new_target, angolazione)
        self.simulate_movement()

        # Si abbassa
        new_target = np.array([posizione_target[0],
                               posizione_target[1],
                               self.obj_size[2] * 0.65])
        self.calculateIK(new_target, angolazione)
        self.simulate_movement()

        # Chiude il gripper
        self.closeGripper()

        # Solleva
        new_target = np.array(self.place_pos)

        # Calcola angoli desiderati robot
        self.calculateIK(new_target, angolazione)
        # Esegue la simulazione
        self.simulate_movement()
        self.openGripper(-0.01)

    def crea_oggetto_l(self, position, orientation):
        """
        Crea un oggetto a forma di L.

        Args:
            position (np.array): nuova posizione dell'oggetto
            orientation (double): nuova orientazione dell'oggetto
        Returns:
            l_shape_id: ID dell'oggetto creato
        """

        # Definisce le dimensioni per i blocchi che costituiranno la forma a L
        block1_size = self.obj_size  # Dimensioni del primo blocco
        block2_size = [block1_size[1], block1_size[0], block1_size[2]]  # Dimensioni del secondo blocco

        # Definisce la posizione di base dell'oggetto a L
        base_position = [0, 0, 0]

        # Definisce le posizioni relative dei blocchi
        block1_position = [0, 0, 0]  # Spostamento del primo blocco
        block2_position = [block1_size[0] - block1_size[1], block1_size[1], 0]  # Spostamento del secondo blocco

        # Crea i "collision shapes" per i due blocchi
        block1_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=block1_size)
        block2_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=block2_size)

        # Crea i "visual shapes" per i due blocchi (opzionale, per la visualizzazione)
        block1_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=block1_size, rgbaColor=[0, 0, 0.5, 1])
        block2_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=block2_size, rgbaColor=[0, 0, 0.5, 1])

        # Definisce la massa dell'oggetto
        mass = 1

        # Crea l'oggetto a forma di L utilizzando due blocchi
        l_shape_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=block1_collision,
            baseVisualShapeIndex=block1_visual,
            basePosition=np.add(base_position, block1_position),
            # Aggiunge il secondo blocco come "link" del primo
            linkMasses=[mass],
            linkCollisionShapeIndices=[block2_collision],
            linkVisualShapeIndices=[block2_visual],
            linkPositions=[block2_position],
            linkOrientations=[p.getQuaternionFromEuler([0, 0, 0])],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],  # Joint fisso per mantenere uniti i blocchi
            linkJointAxis=[[0, 0, 0]]
        )

        # Resetta la posizione dell'oggetto e la sua orientazione
        p.resetBasePositionAndOrientation(l_shape_id, position, orientation)

        return l_shape_id

    def crea_parallelepipedo(self, position, orientation):
        """
        Crea un oggetto a forma di parallelepipedo.

        Args:
            position (np.array): nuova posizione dell'oggetto
            orientation (double): nuova orientazione dell'oggetto
        Returns:
            cubeId: ID dell'oggetto creato
        """
        cubeId = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=self.obj_size),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=self.obj_size, rgbaColor=[0, 0, 0.5, 1]),
            basePosition=position,
            baseOrientation=orientation
        )
        return cubeId

    def crea_oggetto_z(self, position, orientation):
        """
        Crea un oggetto a forma di Z.

        Args:
            position (np.array): nuova posizione dell'oggetto
            orientation (double): nuova orientazione dell'oggetto
        Returns:
            l_shape_id: ID dell'oggetto creato
        """
        # Definisce le dimensioni per i blocchi che costituiranno la forma a Z
        block1_size = self.obj_size*0.6  # Dimensioni del primo blocco
        block1_size[2] = self.obj_size[2]
        block1_size[0] = block1_size[0]*0.8  # Dimensioni del primo blocco
        block2_size = [block1_size[1]*1.33, block1_size[0], block1_size[2]]  # Dimensioni del secondo blocco (diagonale)
        block3_size = block1_size  # Dimensioni del terzo blocco (uguale al primo)

        # Definisce la posizione di base dell'oggetto a Z
        base_position = [0, 0, 0]

        # Definisce le posizioni relative dei blocchi
        block1_position = [0, 0, 0]  # Primo blocco in alto
        block3_position = [-2.625 * block1_size[1], 0, 0]  # Blocco in basso
        block2_position = [block3_position[0]/2, 0, 0]  # Blocco centrale (diagonale)

        # Crea i "collision shapes" per i tre blocchi
        block1_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=block1_size)
        block2_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=block2_size)
        block3_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=block3_size)

        # Creare i "visual shapes" per i tre blocchi (opzionale, per la visualizzazione)
        block1_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=block1_size, rgbaColor=[0, 0, 0.5, 1])
        block2_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=block2_size, rgbaColor=[0, 0, 0.5, 1])
        block3_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=block3_size, rgbaColor=[0, 0, 0.5, 1])

        # Definisce la massa dell'oggetto
        mass = 1

        # Crea l'oggetto a forma di Z utilizzando tre blocchi
        z_shape_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=block1_collision,
            baseVisualShapeIndex=block1_visual,
            basePosition=np.add(base_position, block1_position),
            # Aggiunge il secondo e il terzo blocco come "link" del primo
            linkMasses=[mass, mass],
            linkCollisionShapeIndices=[block2_collision, block3_collision],
            linkVisualShapeIndices=[block2_visual, block3_visual],
            linkPositions=[block2_position, block3_position],
            linkOrientations=[p.getQuaternionFromEuler([0, 0, np.pi/6]), p.getQuaternionFromEuler([0, 0, 0])],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],  # Joint fissi per mantenere uniti i blocchi
            linkJointAxis=[[0, 0, 0], [0, 0, 0]]
        )

        # Questo viene fatto perché altrimenti le forme più complesse escono dal bordo della camera
        position[0] = position[0] * 0.35
        position[1] = position[1] * 0.35

        # Resetta la posizione dell'oggetto e la sua orientazione
        p.resetBasePositionAndOrientation(z_shape_id, position, orientation)

        return z_shape_id
