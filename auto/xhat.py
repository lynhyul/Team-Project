import time

from pyA20.gpio import gpio as GPIO
from pyA20.gpio import port
import threading

class OrangePwm(threading.Thread):

  def __init__(self, frequency, gpioPin, gpioScheme=0):
     """ 
     Init the OrangePwm instance. Expected parameters are :
     - frequency : the frequency in Hz for the PWM pattern. A correct value may be 100.
     - gpioPin : the gpio.port which will act as PWM ouput
     - gpioScheme : saved for compatibility with PiZyPWM code
     """
     self.baseTime = 1.0 / frequency
     self.maxCycle = 100.0
     self.sliceTime = self.baseTime / self.maxCycle
     self.gpioPin = gpioPin
     self.terminated = False
     self.toTerminate = False
     #GPIO.setmode(gpioScheme)


  def start(self, dutyCycle):
    """
    Start PWM output. Expected parameter is :
    - dutyCycle : percentage of a single pattern to set HIGH output on the GPIO pin
    
    Example : with a frequency of 1 Hz, and a duty cycle set to 25, GPIO pin will 
    stay HIGH for 1*(25/100) seconds on HIGH output, and 1*(75/100) seconds on LOW output.
    """
    self.dutyCycle = dutyCycle
    GPIO.setcfg(self.gpioPin, GPIO.OUTPUT)
    self.thread = threading.Thread(None, self.run, None, (), {})
    self.thread.start()


  def run(self):
    """
    Run the PWM pattern into a background thread. This function should not be called outside of this class.
    """
    while self.toTerminate == False:
      if self.dutyCycle > 0:
        GPIO.output(self.gpioPin, GPIO.HIGH)
        time.sleep(self.dutyCycle * self.sliceTime)
      
      if self.dutyCycle < self.maxCycle:
        GPIO.output(self.gpioPin, GPIO.LOW)
        time.sleep((self.maxCycle - self.dutyCycle) * self.sliceTime)

    self.terminated = True


  def ChangeDutyCycle(self, dutyCycle):
    """
    Change the duration of HIGH output of the pattern. Expected parameter is :
    - dutyCycle : percentage of a single pattern to set HIGH output on the GPIO pin
    
    Example : with a frequency of 1 Hz, and a duty cycle set to 25, GPIO pin will 
    stay HIGH for 1*(25/100) seconds on HIGH output, and 1*(75/100) seconds on LOW output.
    """
    self.dutyCycle = dutyCycle


  def changeFrequency(self, frequency):
    """
    Change the frequency of the PWM pattern. Expected parameter is :
    - frequency : the frequency in Hz for the PWM pattern. A correct value may be 100.
    
    Example : with a frequency of 1 Hz, and a duty cycle set to 25, GPIO pin will 
    stay HIGH for 1*(25/100) seconds on HIGH output, and 1*(75/100) seconds on LOW output.
    """
    self.baseTime = 1.0 / frequency
    self.sliceTime = self.baseTime / self.maxCycle


  def stop(self):
    """
    Stops PWM output.
    """
    self.toTerminate = True
    while self.terminated == False:
      # Just wait
      time.sleep(0.01)
  
    GPIO.output(self.gpioPin, GPIO.LOW)
    GPIO.setcfg(self.gpioPin, GPIO.INPUT)
    
    
class L298NMDc(object):
    """ Class to control DC motor via L298n motor controller
    6 methods 1. __init__ 2. forward
    3.backward 4.stop 5 .brake 6.cleanup"""

    def __init__(self, pin_one, pin_two,
                 pwm_pin, freq=50, verbose=False, name="DCMotorX"):
        """ init method
        (1) pin_one, type=int,  GPIO pin connected to IN1 or IN3
        (2) Pin two type=int, GPIO pin connected to IN2 or IN4
        (3) pwm_pin type=int, GPIO pin connected to EnA or ENB
        (4) freq in Hz default 50
        (5) verbose, type=bool  type=bool default=False
         help="Write pin actions"
        (6) name, type=string, name attribute
        """
        self.name = name
        self.pin_one = pin_one
        self.pin_two = pin_two
        self.pwm_pin = pwm_pin
        self.freq = freq
        self.verbose = verbose

        GPIO.init()
        
        GPIO.setcfg(self.pin_one, GPIO.OUTPUT)
        GPIO.setcfg(self.pin_two, GPIO.OUTPUT)
        GPIO.setcfg(self.pwm_pin, GPIO.OUTPUT)

        self.my_pwm = OrangePwm(100, self.pwm_pin)
        
        self.last_pwm = 0
        self.my_pwm.start(self.last_pwm)
        if self.verbose:
            print(" Motor initialized named: {} ".format(self.name))
            print(" Pin one In1 or In3:  {}".format(self.pin_one))
            print(" Pin two In2 or in4:  {}".format(self.pin_two))
            print(" Pin pwm enA or enB:  {}".format(self.pwm_pin))
            print(" Frequency: {} ".format(self.freq))

    def forward(self, duty_cycle=50):
        """ Move motor forwards passed duty cycle for speed control """
        GPIO.output(self.pin_one, True)
        GPIO.output(self.pin_two, False)
        if self.verbose:
            print("Moving Motor Forward : Duty Cycle = {}".format(duty_cycle))
        if duty_cycle != self.last_pwm:
            self.my_pwm.ChangeDutyCycle(duty_cycle)
            self.last_pwm = duty_cycle

    def backward(self, duty_cycle=50):
        """ Move motor backwards passed duty cycle for speed control"""
        GPIO.output(self.pin_one, False)
        GPIO.output(self.pin_two, True)
        if self.verbose:
            print("Moving Motor Backward : Duty Cycle = {}".format(duty_cycle))
        if duty_cycle != self.last_pwm:
            self.my_pwm.ChangeDutyCycle(duty_cycle)
            self.last_pwm = duty_cycle

    def stop(self, duty_cycle=0):
        """ Stop motor"""
        GPIO.output(self.pin_one, False)
        GPIO.output(self.pin_two, False)
        if self.verbose:
            print("Stoping Motor : Duty Cycle = {}".format(duty_cycle))
        if duty_cycle != self.last_pwm:
            self.my_pwm.ChangeDutyCycle(duty_cycle)
            self.last_pwm = duty_cycle
        

    def brake(self, duty_cycle=100):
        """ brake motor"""
        GPIO.output(self.pin_one, True)
        GPIO.output(self.pin_two, True)
        if self.verbose:
            print("Braking Motor : Duty Cycle = {}".format(duty_cycle))
        if duty_cycle != self.last_pwm:
            self.my_pwm.ChangeDutyCycle(duty_cycle)
            self.last_pwm = duty_cycle

    def cleanup(self, clean_up=False):
        """ cleanup all GPIO connections used in event of error by lib user"""
        if self.verbose:
            print("rpi_dc_lib.py : Cleaning up")
        GPIO.output(self.pin_one, False)
        GPIO.output(self.pin_two, False)
        self.my_pwm.ChangeDutyCycle(0)
        self.my_pwm.stop()
        #if clean_up:
        #    GPIO.cleanup()




# define instance of the class 
# (GPIO , GPIO , GPIO , freq , verbose, name)
#MotorOne = rpi_dc_lib.L298NMDc(20 ,21 ,16 ,50 ,False, "motor_one")
#MotorTwo = rpi_dc_lib.L298NMDc(13 ,19 ,26 ,50 ,False, "motor_two")
###MotorOne = L298NMDc(port.PG6 ,port.PG7 ,port.PG9 ,50 ,False, "motor_one")
###MotorTwo = L298NMDc(port.PA9 ,port.PA10 ,port.PA20 ,50 ,False, "motor_two")

MotorOne = L298NMDc(port.PG6 ,port.PG9 ,port.PG7 ,50 ,False, "motor_one")
MotorTwo = L298NMDc(port.PG8 ,port.PA18 ,port.PA21 ,50 ,False, "motor_two")


def motor_one_speed(speed):
    try:
        if speed < 0:
            MotorOne.backward(-speed)
        elif speed > 0:
            MotorOne.forward(speed)
        else:
            MotorOne.stop(0)
            #print("motor stopped\n")
            
      
    except KeyboardInterrupt:
            print("CTRL-C: Terminating program.")
            MotorOne.cleanup(True)
    except Exception as error:
            print(error)
            print("Unexpected error:")
            MotorOne.cleanup(True)
    #else:
    #    #print("No errors")
    #finally:
    #    #print("cleaning up")
    #    #MotorOne.cleanup(True)

def motor_two_speed(speed):
    try:
        if speed < 0:
            MotorTwo.backward(-speed)
        elif speed > 0:
            MotorTwo.forward(speed)
        else:
            MotorTwo.stop(0)
            #print("motor stopped\n")
            
    except KeyboardInterrupt:
            print("CTRL-C: Terminating program.")
            MotorTwo.cleanup(True)
    except Exception as error:
            print(error)
            print("Unexpected error:")
            MotorTwo.cleanup(True)
    #else:
    #    #print("No errors")
    #finally:
    #    #print("cleaning up")
    #    #MotorTwo.cleanup(True)

def motor_clean():
   MotorOne.stop(0)
   MotorTwo.stop(0)
   MotorOne.cleanup(True)
   MotorTwo.cleanup(True)
    
if __name__ == '__main__':
    while(True):
        motor_one_speed(30)
        motor_two_speed(30)
        time.sleep(3)
        motor_one_speed(0)
        motor_two_speed(0)
        time.sleep(1)
        motor_one_speed(-30)
        motor_two_speed(-30)
        time.sleep(3)
        motor_clean()
        break
    exit()