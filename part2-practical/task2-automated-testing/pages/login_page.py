"""
Page Object Model for Login Page
"""

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import logging

logger = logging.getLogger(__name__)

class LoginPage:
    """Page Object Model for login functionality."""
    
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(driver, 10)
        
    # Locators
    USERNAME_FIELD = (By.ID, "username")
    PASSWORD_FIELD = (By.ID, "password")
    SUBMIT_BUTTON = (By.ID, "submit")
    ERROR_MESSAGE = (By.ID, "error")
    SUCCESS_MESSAGE = (By.CLASS_NAME, "post-title")
    
    def enter_username(self, username: str):
        """Enter username in the username field."""
        try:
            element = self.wait.until(EC.presence_of_element_located(self.USERNAME_FIELD))
            element.clear()
            element.send_keys(username)
            logger.info(f"Entered username: {username}")
        except TimeoutException:
            logger.error("Username field not found")
            raise
    
    def enter_password(self, password: str):
        """Enter password in the password field."""
        try:
            element = self.wait.until(EC.presence_of_element_located(self.PASSWORD_FIELD))
            element.clear()
            element.send_keys(password)
            logger.info("Password entered")
        except TimeoutException:
            logger.error("Password field not found")
            raise
    
    def click_submit(self):
        """Click the submit button."""
        try:
            element = self.wait.until(EC.element_to_be_clickable(self.SUBMIT_BUTTON))
            element.click()
            logger.info("Submit button clicked")
        except TimeoutException:
            logger.error("Submit button not clickable")
            raise
    
    def get_error_message(self) -> str:
        """Get error message text."""
        try:
            element = self.wait.until(EC.presence_of_element_located(self.ERROR_MESSAGE))
            return element.text
        except TimeoutException:
            return ""
    
    def is_login_successful(self) -> bool:
        """Check if login was successful."""
        try:
            self.wait.until(EC.presence_of_element_located(self.SUCCESS_MESSAGE))
            return True
        except TimeoutException:
            return False
    
    def login(self, username: str, password: str):
        """Complete login process."""
        self.enter_username(username)
        self.enter_password(password)
        self.click_submit()