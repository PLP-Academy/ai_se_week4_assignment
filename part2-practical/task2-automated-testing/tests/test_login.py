"""
Automated login tests using Selenium WebDriver
"""

import pytest
import time
from pages.login_page import LoginPage

class TestLogin:
    """Test suite for login functionality."""
    
    BASE_URL = "https://practicetestautomation.com/practice-test-login/"
    
    def setup_method(self):
        """Setup before each test method."""
        self.results = []
    
    @pytest.mark.usefixtures("screenshot_on_failure")
    def test_valid_login(self, driver):
        """Test login with valid credentials."""
        driver.get(self.BASE_URL)
        login_page = LoginPage(driver)
        
        # Valid credentials
        login_page.login("student", "Password123")
        
        # Verify successful login
        assert login_page.is_login_successful(), "Login should be successful with valid credentials"
        assert "Logged In Successfully" in driver.page_source
        
        self.results.append({"test": "valid_login", "status": "PASS"})
    
    @pytest.mark.usefixtures("screenshot_on_failure")
    def test_invalid_password(self, driver):
        """Test login with invalid password."""
        driver.get(self.BASE_URL)
        login_page = LoginPage(driver)
        
        # Invalid password
        login_page.login("student", "wrongpassword")
        
        # Verify error message
        error_message = login_page.get_error_message()
        assert "Your password is invalid!" in error_message
        assert not login_page.is_login_successful()
        
        self.results.append({"test": "invalid_password", "status": "PASS"})
    
    @pytest.mark.usefixtures("screenshot_on_failure")
    def test_invalid_username(self, driver):
        """Test login with invalid username."""
        driver.get(self.BASE_URL)
        login_page = LoginPage(driver)
        
        # Invalid username
        login_page.login("invaliduser", "Password123")
        
        # Verify error message
        error_message = login_page.get_error_message()
        assert "Your username is invalid!" in error_message
        assert not login_page.is_login_successful()
        
        self.results.append({"test": "invalid_username", "status": "PASS"})
    
    @pytest.mark.usefixtures("screenshot_on_failure")
    def test_empty_fields(self, driver):
        """Test login with empty fields."""
        driver.get(self.BASE_URL)
        login_page = LoginPage(driver)
        
        # Empty credentials
        login_page.login("", "")
        
        # Verify error message
        error_message = login_page.get_error_message()
        assert "Your username is invalid!" in error_message
        assert not login_page.is_login_successful()
        
        self.results.append({"test": "empty_fields", "status": "PASS"})