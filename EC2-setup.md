#install apache
#run letsencrypt for apache

a2enmod http_proxy

#modify /etc/apache2/sites-available/000-default-l1-ssl.conf: 
    add to <VirtualHost *:443>
        ProxyPass / http://localhost:5000/
        
sudo service apache2 restart

install python3 pip
pip install flask gunicorn 

gunicorn --bind 127.0.0.1:5000 wsgi
