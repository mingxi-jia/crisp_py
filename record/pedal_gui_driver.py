import hid
import time
import tkinter as tk
import sys
import threading
import queue # For thread-safe communication

# --- Default Configuration ---
# These can be overridden during class instantiation
DEFAULT_CONFIG = {
    "VENDOR_ID": 0x3553,
    "PRODUCT_ID": 0xb001,
    "REPORT_SIZE": 8,
    "STATE_WAITING": "Waiting",
    "STATE_RECORDING": "Recording",
    "ACTION_DELETE": "Delete",
    "ACTION_SAVE": "Save",
    "COLOR_WAITING": "red",
    "COLOR_RECORDING": "green",
    "COLOR_ACTION_FG": "yellow",
    "COLOR_STATE_FG": "white",
    "WINDOW_TITLE": "Pedal Status",
    "WINDOW_GEOMETRY": "300x150",
    "POLL_INTERVAL_MS": 20,       # How often the HID device is polled
    "QUEUE_CHECK_INTERVAL_MS": 50, # How often the GUI checks the event queue
    "ACTION_DISPLAY_MS": 1000,
    "VALUE_DELETE": 6,
    "VALUE_SAVE": 4,
    "VALUE_RECORDING_START": 5,
    "VALUE_RELEASED": 0,
    # Optional Callbacks for external integration
    "on_delete_callback": None,
    "on_save_callback": None,
    "on_record_start_callback": None,
    "on_record_stop_callback": None,
    "on_state_change_callback": None, # Called with new state (Waiting/Recording)
}

class PedalListenerGUI:
    """
    A class to listen to a USB HID foot pedal, display status in a Tkinter GUI,
    and handle specific pedal actions (Delete, Save, Record Start/Stop).

    Uses a queue for thread-safe communication between HID polling and GUI updates.
    Can be run standalone or imported as a module.
    """
    def __init__(self, config=None, run_in_thread=False):
        """
        Initializes the PedalListenerGUI.

        Args:
            config (dict, optional): A dictionary overriding default configuration values.
                                     Defaults to DEFAULT_CONFIG.
            run_in_thread (bool): If True, HID polling runs in a separate thread.
                                  If False (default), HID polling is scheduled via root.after.
        """
        # Merge provided config with defaults
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # --- Instance Variables ---
        self.h = None # HID device handle
        self.root = None # Tkinter root window
        self.status_label = None # Tkinter label for status display
        self.current_state = self.config["STATE_WAITING"] # Initial state
        self.last_data = None # Last read HID report (used by polling logic)
        self.action_message_job = None # Tkinter 'after' job ID for temporary messages
        self.queue_check_job = None # Tkinter 'after' job ID for queue processing
        self.is_running = threading.Event() # Use Event for thread-safe running flag
        self.run_hid_in_thread = run_in_thread # Distinguish from GUI thread flag
        self.hid_thread = None # Thread object for HID polling if run_hid_in_thread is True
        self.event_queue = queue.Queue() # Thread-safe queue for events

    # --- GUI Update Methods (MUST run in main thread) ---

    def _update_gui_state(self):
        """Internal method to update the GUI window color and text based ONLY on the current_state."""
        if not self.root: return

        # Cancel any pending job to revert an action message
        if self.action_message_job:
            try:
                self.root.after_cancel(self.action_message_job)
            except tk.TclError: pass # Ignore if window closing
            self.action_message_job = None

        color = self.config["COLOR_WAITING"]
        status_text = self.config["STATE_WAITING"]

        if self.current_state == self.config["STATE_RECORDING"]:
            color = self.config["COLOR_RECORDING"]
            status_text = self.config["STATE_RECORDING"]

        try:
            self.root.config(bg=color)
            if self.status_label:
                self.status_label.config(text=status_text, bg=color, fg=self.config["COLOR_STATE_FG"])
        except tk.TclError:
            # This might happen if called during shutdown
            print("GUI update skipped, window might be closing.", file=sys.stderr)

    def _show_temporary_message(self, message):
        """Internal method to display a temporary message (Delete/Save) on the label."""
        if not self.root or not self.status_label: return

        # Cancel any previous pending revert job
        if self.action_message_job:
            try:
                self.root.after_cancel(self.action_message_job)
            except tk.TclError: pass
            self.action_message_job = None

        try:
            current_bg = self.root.cget("bg") # Keep current background
            self.status_label.config(text=message, fg=self.config["COLOR_ACTION_FG"], bg=current_bg)
            # Schedule revert back to persistent state display
            self.action_message_job = self.root.after(self.config["ACTION_DISPLAY_MS"], self._update_gui_state)
        except tk.TclError:
            print("Failed to show temporary message, window might be closing.", file=sys.stderr)

    # --- Event Queue Processing (Runs in main thread via root.after) ---

    def _process_event_queue(self):
        """Checks the event queue and processes events in the main GUI thread."""
        if not self.is_running.is_set() or not self.root:
             # Stop checking if not running or GUI is gone
             return

        try:
            while not self.event_queue.empty():
                try:
                    event_type, data = self.event_queue.get_nowait()

                    if event_type == "HID_DATA":
                        self._handle_hid_data(data)
                    elif event_type == "ERROR":
                        print(f"Error reported from HID thread: {data}", file=sys.stderr)
                        # Optionally display error in GUI or trigger stop
                        self.stop() # Stop on HID error
                    # Add other event types if needed

                except queue.Empty:
                    break # No more events for now
                except Exception as e:
                     print(f"Error processing event queue item: {e}", file=sys.stderr)

            # Schedule the next check
            self.queue_check_job = self.root.after(self.config["QUEUE_CHECK_INTERVAL_MS"], self._process_event_queue)

        except tk.TclError:
            # Window likely destroyed
            print("Queue processing stopped due to TclError (window closed).", file=sys.stderr)
            self.is_running.clear() # Ensure polling stops
        except Exception as e:
            print(f"Unexpected error in _process_event_queue: {e}", file=sys.stderr)
            self.stop() # Attempt cleanup

    def _handle_hid_data(self, d):
        """Processes decoded HID data - runs in the main GUI thread."""
        if not d: return # Ignore empty data

        print(f"Processing data: {d}") # Debug print

        # --- Decoding Logic (now in main thread) ---
        if len(d) > 3: # Ensure relevant byte exists
            pedal_value = d[3] # Extract pedal status value

            state_changed = False
            new_state = self.current_state # Assume no state change initially

            callback_to_run = None # Store callback to run after GUI updates

            if pedal_value == self.config["VALUE_DELETE"]:
                self._show_temporary_message(self.config["ACTION_DELETE"])
                print(self.config["ACTION_DELETE"]) # Console print
                callback_to_run = self.config.get("on_delete_callback")

            elif pedal_value == self.config["VALUE_SAVE"]:
                self._show_temporary_message(self.config["ACTION_SAVE"])
                print(self.config["ACTION_SAVE"]) # Console print
                callback_to_run = self.config.get("on_save_callback")

            elif pedal_value == self.config["VALUE_RECORDING_START"]:
                if self.current_state != self.config["STATE_RECORDING"]:
                    print("State changed to: Recording")
                    new_state = self.config["STATE_RECORDING"]
                    state_changed = True
                    callback_to_run = self.config.get("on_record_start_callback")

            elif pedal_value == self.config["VALUE_RELEASED"]:
                if self.current_state != self.config["STATE_WAITING"]:
                    print("State changed to: Waiting")
                    new_state = self.config["STATE_WAITING"]
                    state_changed = True
                    callback_to_run = self.config.get("on_record_stop_callback") # Trigger stop on release

            # Update state and GUI if changed
            if state_changed:
                self.current_state = new_state
                self._update_gui_state() # Update persistent state display
                # Trigger general state change callback if defined
                state_change_cb = self.config.get("on_state_change_callback")
                if state_change_cb:
                    try:
                        state_change_cb(self.current_state)
                    except Exception as cb_e:
                        print(f"Error in on_state_change_callback: {cb_e}", file=sys.stderr)

            # Execute specific action callback (if any)
            if callback_to_run:
                try:
                    callback_to_run()
                except Exception as cb_e:
                    print(f"Error in action callback: {cb_e}", file=sys.stderr)
        # --- End Decoding ---


    # --- HID Polling (Runs in separate thread OR via root.after) ---

    def _poll_hid_device(self):
        """Reads from HID device and puts data/errors onto the event queue."""
        if not self.h:
            print("HID device not available for polling.", file=sys.stderr)
            return # Should not happen if start() succeeded

        try:
            # Read data (non-blocking)
            d = self.h.read(self.config["REPORT_SIZE"])

            # Process data only if it's new
            if d and list(d) != self.last_data:
                self.last_data = list(d) # Store new data
                self.event_queue.put(("HID_DATA", d)) # Put raw data onto queue

            # If not running in a thread, schedule the next poll
            if not self.run_hid_in_thread and self.is_running.is_set() and self.root:
                 # Use root.after ONLY if polling is happening in the main thread
                 self.root.after(self.config["POLL_INTERVAL_MS"], self._poll_hid_device)

        # except hid.HIDException as e:
        #     print(f"\nHID Error during read: {e}", file=sys.stderr)
        #     self.event_queue.put(("ERROR", str(e))) # Report error via queue
        #     self.is_running.clear() # Signal loop to stop
        except Exception as e:
            print(f"\nUnexpected error during HID polling: {e}", file=sys.stderr)
            self.event_queue.put(("ERROR", str(e))) # Report error via queue
            self.is_running.clear() # Signal loop to stop


    def _poll_hid_thread_target(self):
        """Target function for the background HID polling thread."""
        print("HID polling thread started.")
        while self.is_running.is_set():
             self._poll_hid_device()
             # Add a small sleep to prevent busy-waiting in the thread
             time.sleep(self.config["POLL_INTERVAL_MS"] / 1000.0)
        print("HID polling thread finished.")


    # --- GUI Setup and Control ---

    def _setup_gui(self):
        """Internal method to create and configure the Tkinter window."""
        if self.root: # Avoid creating multiple windows
             print("GUI setup called but window already exists.", file=sys.stderr)
             return False # Indicate GUI was not set up now

        print("Setting up GUI...")
        self.root = tk.Tk()
        self.root.title(self.config["WINDOW_TITLE"])
        self.root.geometry(self.config["WINDOW_GEOMETRY"])

        self.status_label = tk.Label(self.root, text=self.current_state, font=("Helvetica", 16, "bold"), fg=self.config["COLOR_STATE_FG"])
        self.status_label.pack(expand=True, fill="both")

        self._update_gui_state() # Set initial colors/text

        # Register the closing function for graceful shutdown
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        print("GUI setup complete.")
        return True # Indicate GUI was set up

    def _run_gui_loop(self):
        """Internal method to start the Tkinter main loop and queue checking."""
        if not self.root:
             print("GUI not set up, cannot run main loop.", file=sys.stderr)
             return
        try:
            print("Starting Tkinter mainloop and event queue processing...")
            # Start checking the event queue
            self.queue_check_job = self.root.after(self.config["QUEUE_CHECK_INTERVAL_MS"], self._process_event_queue)
            # Start the main GUI event loop (blocks here)
            self.root.mainloop()
            print("Tkinter mainloop finished.")
        except Exception as e:
            print(f"Error during mainloop: {e}", file=sys.stderr)
        finally:
            # Ensure cleanup happens even if mainloop crashes or exits unexpectedly
            print("Mainloop finished or errored, ensuring stop is called...")
            self.stop()


    def start(self):
        """Connects to the HID device, starts the GUI, and begins polling/queue processing."""
        if self.is_running.is_set():
            print("Listener is already running.", file=sys.stderr)
            return

        print("Starting Pedal Listener...")
        # Set running flag early
        self.is_running.set()
        self.last_data = None # Reset last data on start

        try:
            # --- Connect to HID ---
            print(f"Attempting to connect to HID: VID={hex(self.config['VENDOR_ID'])}, PID={hex(self.config['PRODUCT_ID'])}")
            self.h = hid.device()
            self.h.open(self.config["VENDOR_ID"], self.config["PRODUCT_ID"])
            print("HID device opened successfully!")
            try:
                 print("Manufacturer:", self.h.get_manufacturer_string())
                 print("Product:", self.h.get_product_string())
            except Exception: pass # Ignore if descriptors can't be read
            self.h.set_nonblocking(1) # Essential

            # --- Setup GUI (Must happen in main thread) ---
            if not self._setup_gui():
                 # Should not happen unless start is called multiple times concurrently
                 raise RuntimeError("Failed to set up GUI.")

            # --- Start HID Polling ---
            if self.run_hid_in_thread:
                print("Starting HID polling in a separate thread.")
                self.hid_thread = threading.Thread(target=self._poll_hid_thread_target, daemon=True)
                self.hid_thread.start()
            else:
                # Start polling via root.after if not using a separate thread
                print("Starting HID polling via root.after in the main thread.")
                self.root.after(self.config["POLL_INTERVAL_MS"], self._poll_hid_device)

            # --- Start GUI Main Loop (blocks if not run_in_thread was specified for GUI) ---
            # NOTE: This class now assumes the GUI always runs in the thread that calls start().
            # The 'run_in_thread' parameter now only controls the HID polling.
            self._run_gui_loop()


        # except hid.HIDException as e:
        #     print(f"\nFailed to open HID device: {e}", file=sys.stderr)
        #     print("Troubleshooting: Check VID/PID, connection, permissions (udev rules).", file=sys.stderr)
        #     self.h = None # Ensure handle is None if open failed
        #     self.stop() # Cleanup any partial setup
        except Exception as e:
            print(f"\nAn unexpected error occurred during startup: {e}", file=sys.stderr)
            self.stop() # Cleanup

    def stop(self):
        """Stops the listener, closes the HID device, and destroys the GUI window."""
        if not self.is_running.is_set() and not self.root and not self.h:
             # Avoid redundant closing messages if already stopped cleanly
             # print("Listener already stopped or not started.")
             return

        print("\nStopping listener and closing resources...")
        self.is_running.clear() # Signal loops/threads to stop

        # --- Stop HID Thread (if running) ---
        if self.hid_thread and self.hid_thread.is_alive():
             print("Waiting for HID polling thread to join...")
             self.hid_thread.join(timeout=1.0) # Wait briefly for thread to exit
             if self.hid_thread.is_alive():
                  print("Warning: HID thread did not exit cleanly.", file=sys.stderr)
        self.hid_thread = None

        # --- Stop Tkinter ---
        # Needs to happen from the main thread if possible, but destroying
        # the window often triggers the necessary cleanup via WM_DELETE_WINDOW.
        # If stop() is called from another thread, destroying might fail.
        if self.root:
            try:
                # Cancel pending jobs (best effort)
                if self.queue_check_job:
                    self.root.after_cancel(self.queue_check_job)
                    self.queue_check_job = None
                if self.action_message_job:
                    self.root.after_cancel(self.action_message_job)
                    self.action_message_job = None

                # Destroy window (this also breaks the mainloop if it's running)
                print("Destroying Tkinter window...")
                self.root.destroy()
                print("GUI window closed.")
            except tk.TclError as e:
                 # Expected if called from wrong thread or already destroyed
                 print(f"TclError during Tkinter cleanup (normal if stopping from wrong thread): {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error closing Tk window: {e}", file=sys.stderr)
            finally:
                 self.root = None # Mark as destroyed
                 self.status_label = None

        # --- Close HID Device ---
        if self.h:
            try:
                print("Closing HID device...")
                self.h.close()
                print("HID device closed.")
            except Exception as e:
                print(f"Error closing HID device: {e}", file=sys.stderr)
            finally:
                 self.h = None # Mark as closed

        # --- Clear Queue (optional, helps GC) ---
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break
        print("Event queue cleared.")

        print("Listener stopped.")


# --- Main Execution Block (for running standalone) ---
if __name__ == "__main__":

    # --- Example Usage ---
    listener = None # Initialize listener variable

    def my_delete_action():
        print("--- Custom Delete Action Triggered! ---")

    def my_save_action():
        print("--- Custom Save Action Triggered! ---")

    def my_state_change(new_state):
         print(f"--- Custom State Change Detected: {new_state} ---")


    # Example configuration override with callbacks
    custom_config = {
        "on_delete_callback": my_delete_action,
        "on_save_callback": my_save_action,
        "on_state_change_callback": my_state_change,
        # "VALUE_DELETE": 8 # Example: if your delete button sends '8'
    }

    try:
        # Instantiate the class with custom config
        # Set run_hid_in_thread=True to test the threading fix.
        # The GUI itself always runs in the main thread now.
        print("Running example with HID polling in background thread.")
        listener = PedalListenerGUI(config=custom_config, run_in_thread=True)

        # Start the listener (this will block until the GUI is closed)
        listener.start()

    except KeyboardInterrupt:
        print("\nCtrl+C detected in main block.")
        # If listener is running, stop() should be called automatically
        # when the mainloop exits or WM_DELETE_WINDOW is handled.
        # Explicitly calling stop here might be needed if start() failed partially.
        if listener and listener.is_running.is_set():
            print("Attempting explicit stop from KeyboardInterrupt...")
            listener.stop()
    except Exception as e:
        print(f"\nAn error occurred in the main execution block: {e}", file=sys.stderr)
    finally:
        # Ensure stop is called if listener was initialized but failed/interrupted
        if listener and listener.is_running.is_set():
             print("Ensuring listener is stopped from main finally block...")
             listener.stop()
        print("Main script execution finished.")

    # --- Example of Importing (in another script) ---
    # from your_module_name import PedalListenerGUI
    # import time
    #
    # def handle_record_start():
    #     print("Now recording data...")
    #
    # config = {"on_record_start_callback": handle_record_start}
    #
    # # Run HID polling in thread so your main script can continue
    # # The PedalListenerGUI object itself needs to be managed by the main script.
    # # The start() method will still block if called from the main thread,
    # # because the GUI mainloop runs there.
    # # To truly run independently, the main script would need to manage
    # # the PedalListenerGUI instance in its *own* thread, separate from its
    # # primary work. This adds complexity.
    #
    # # Simpler approach: Run HID in thread, call start() from main thread.
    # print("Starting pedal monitor with HID polling in background thread...")
    # pedal_monitor = PedalListenerGUI(config=config, run_in_thread=True)
    #
    # # If you need the main script to do other things *while* the GUI is open,
    # # you would need to run pedal_monitor.start() in yet another thread.
    # # For example:
    # # monitor_thread = threading.Thread(target=pedal_monitor.start, daemon=True)
    # # monitor_thread.start()
    # # print("Pedal monitor GUI started in its own thread.")
    # # time.sleep(30) # Example: run for 30 seconds
    # # print("Stopping pedal monitor.")
    # # pedal_monitor.stop() # Request stop
    # # monitor_thread.join() # Wait for GUI thread to finish cleanup
    #
    # # If blocking the main script until GUI closes is acceptable:
    # pedal_monitor.start() # This blocks until GUI is closed.
    #
    # print("Pedal monitor finished.")

