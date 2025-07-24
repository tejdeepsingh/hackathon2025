FROM andrius/asterisk

# Copy custom config files
COPY pjsip.conf /etc/asterisk/pjsip.conf
COPY extensions.conf /etc/asterisk/extensions.conf

# Copy the custom script
COPY cdr_logs.sh /usr/local/bin/cdr_logs.sh

COPY beep.wav /var/lib/asterisk/sounds
COPY welcome1.wav /var/lib/asterisk/sounds
COPY ivr-options.wav /var/lib/asterisk/sounds
COPY please-wait-connect-call.wav /var/lib/asterisk/sounds

# Make the script executable
RUN chmod +x /usr/local/bin/cdr_logs.sh


# Or, run it during container startup
CMD /usr/local/bin/cdr_logs.sh && /run.sh

