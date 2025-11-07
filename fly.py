import logging
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='udp://192.168.43.42:2390')

logging.basicConfig(level=logging.ERROR)

def run_motor_test(scf):
    """
    Arms, ramps motors, and disarms.
    """
    commander = scf.cf.commander
    platform = scf.cf.platform

    # Arming Step
    print('Arming the drone...')
    platform.send_arming_request(True)
    time.sleep(0.5)
    
    # Unlock Commander
    print('Unlocking commander...')
    commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)
    
    # Thrust values
    TEST_THRUST = 38000 
    
    print('--- HOLD THE DRONE FIRMLY NOW ---')
    print(f'Ramping up thrust to {TEST_THRUST} over 2 seconds...')
    
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > 2.0:
            break
        
        ramp_ratio = elapsed / 2.0
        current_thrust = int(10001 + (TEST_THRUST - 10001) * ramp_ratio)
        
        commander.send_setpoint(0, 0, 0, current_thrust)
        time.sleep(0.02) 
        
    print(f'Holding at {TEST_THRUST} for 3 seconds...')
    commander.send_setpoint(0, 0, 0, TEST_THRUST)
    time.sleep(3)
    
    print('Ramping down to 0...')
    
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            break
        
        ramp_ratio = 1.0 - (elapsed / 1.0)
        current_thrust = int(10001 + (TEST_THRUST - 10001) * ramp_ratio)
        
        commander.send_setpoint(0, 0, 0, current_thrust)
        time.sleep(0.02)
        
    print('Shutting off motors.')
    commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)

    # Disarming Step
    print('Disarming the drone...')
    platform.send_arming_request(False)
    time.sleep(0.1)


if __name__ == '__main__':
    cflib.crtp.init_drivers()

    print(f'Connecting to {URI}...')
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        print('Connected!')
        
        param_name = 'stabilizer.estimator'
        
        # --- NEW: FORCING BASIC ESTIMATOR ---
        print('Waiting for parameters to load...')
        start_time = time.time()
        while time.time() - start_time < 5.0:  # 5 second timeout
            try:
                # Try to get the value
                val = scf.cf.param.get_value(param_name)
                # If it succeeds, break the loop
                break
            except KeyError:
                # If it fails (KeyError), it's not loaded yet.
                time.sleep(0.1)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                sys.exit(1)
        
        if val is None:
            print("Error: Timed out waiting for 'stabilizer.estimator' param.")
            sys.exit(1)

        print(f'Current estimator value: {val}')
        if int(val) != 1:
            print('Setting estimator to 1 (Complementary filter)...')
            scf.cf.param.set_value(param_name, 1)
            time.sleep(0.5) # Give it time to set
        else:
            print('Estimator is already set to 1.')
        # ------------------------------------
        
        # Run the safe motor test
        run_motor_test(scf)

    print('Script finished.')
