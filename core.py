"""Concurrent testing during training."""
from collections import *
import platform
import threading
import time
import traceback
import numpy as np
import tensorflow as tf
import logging
import os
import re
import sys
from typing import Any, Dict, List
import multiprocessing
import queue
import matplotlib.pyplot as plt
from tensorflow.python.distribute.device_util import current

logger = logging.getLogger(__name__)


class LiveTester(object):
    """Manage concurrent testing on test data source."""

    def __init__(self, model, data_source, use_batch_statistics=True):
        """Initialize tester with reference to model and data sources."""
        self.model = model
        self.data = data_source
        self.time = self.model.time
        self.summary = self.model.summary
        self._tensorflow_session = model._tensorflow_session

        self._is_testing = False
        self._condition = threading.Condition()

        self._use_batch_statistics = use_batch_statistics

    def stop(self):
        logger.info('LiveTester::stop is being called.')
        self._is_testing = False

    def __del__(self):
        """Handle deletion of instance by closing thread."""
        if not hasattr(self, '_coordinator'):
            return
        self._coordinator.request_stop()
        with self._condition:
            self._is_testing = True  # Break wait if waiting
            self._condition.notify_all()
        self._coordinator.join([self._thread], stop_grace_period_secs=1)

    def _true_if_testing(self):
        return self._is_testing

    def trigger_test_if_not_testing(self, current_step):
        """If not currently testing, run test."""
        if not self._is_testing:
            with self._condition:
                self._is_testing = True
                self._testing_at_step = current_step
                self._condition.notify_all()

    def test_job(self):
        """Evaluate requested metric over entire test set."""
        while not self._coordinator.should_stop():
            with self._condition:
                self._condition.wait_for(self._true_if_testing)
                if self._coordinator.should_stop():
                    break
                should_stop = False
                try:
                    should_stop = self.do_full_test()
                except:
                    traceback.print_exc()
                self._is_testing = False
                if should_stop is True:
                    break
        logger.debug('Exiting thread %s' % threading.current_thread().name)

    def do_full_test(self, sleep_between_batches=0.2):
        # Copy current weights over
        self.copy_model_weights()

        # Reset data sources
        for data_source_name, data_source in self.data.items():
            data_source.reset()
            num_batches = int(data_source.num_entries / data_source.batch_size)

        # Decide what to evaluate
        fetches = self._tensors_to_evaluate
        outputs = dict([(name, list()) for name in fetches.keys()])

        # Select random index to produce (image) summaries at
        summary_index = np.random.randint(num_batches)

        self.time.start('full test')
        for i in range(num_batches):
            if self._is_testing is not True:
                logger.debug('Testing flag found to be `False` at iter. %d' % i)
                break
            logger.debug('Testing on %03d/%03d batches.' % (i + 1, num_batches))
            if i == summary_index:
                fetches['summaries'] = self.summary.get_ops(mode='test')
            try:
                output = self._tensorflow_session.run(
                    fetches=fetches,
                    feed_dict={
                        self.model.is_training: False,
                        self.model.use_batch_statistics: self._use_batch_statistics,
                    },
                )
            except (tf.errors.CancelledError, RuntimeError):
                return True
            time.sleep(sleep_between_batches)  # Brief pause to prioritise training
            if 'summaries' in output:  # Write summaries on first batch
                self.summary.write_summaries(output['summaries'], self._testing_at_step)
                del fetches['summaries']
                del output['summaries']
            for name, value in output.items():  # Gather results from this batch
                outputs[name].append(output[name])
        self.time.end('full test')

        # If incomplete, skip this round of tests (most likely shutting down)
        if len(list(outputs.values())[0]) != num_batches:
            return True

        # Calculate mean values
        for name, values in outputs.items():
            outputs[name] = np.mean(values)

        # TODO: Log metric as summary
        to_print = '[Test at step %06d] ' % self._testing_at_step
        to_print += ', '.join([
            '%s = %f' % (name, value) for name, value in outputs.items()
        ])
        logger.info(to_print)

        # Store mean metrics/losses (and other summaries)
        feed_dict = dict([(self._placeholders[name], value)
                         for name, value in outputs.items()])
        feed_dict[self.model.is_training] = False
        feed_dict[self.model.use_batch_statistics] = True
        try:
            summaries = self._tensorflow_session.run(
                fetches=self.summary.get_ops(mode='full_test'),
                feed_dict=feed_dict,
            )
        except (tf.errors.CancelledError, RuntimeError):
            return True
        self.summary.write_summaries(summaries, self._testing_at_step)

        return False

    def do_final_full_test(self, current_step):
        logger.info('Stopping the live testing threads.')

        # Stop thread(s)
        self._is_testing = False
        self._coordinator.request_stop()
        with self._condition:
            self._is_testing = True  # Break wait if waiting
            self._condition.notify_all()
        self._coordinator.join([self._thread], stop_grace_period_secs=1)

        # Start final full test
        logger.info('Running final full test')
        self.copy_model_weights()
        self._is_testing = True
        self._testing_at_step = current_step
        self.do_full_test(sleep_between_batches=0)

    def _post_model_build(self):
        """Prepare combined operation to copy model parameters over from CPU/GPU to CPU."""
        with tf.variable_scope('copy2test'):
            all_variables = tf.global_variables()
            train_vars = dict([(v.name, v) for v in all_variables
                               if not v.name.startswith('test/')])
            test_vars = dict([(v.name, v) for v in all_variables
                              if v.name.startswith('test/')])
            self._copy_variables_to_test_model_op = tf.tuple([
                test_vars['test/' + k].assign(train_vars[k]) for k in train_vars.keys()
                if 'test/' + k in test_vars
            ])

        # Begin testing thread
        self._coordinator = tf.train.Coordinator()
        self._thread = threading.Thread(target=self.test_job,
                                        name='%s_tester' % self.model.identifier)
        self._thread.daemon = True
        self._thread.start()

        # Pick tensors we need to evaluate
        all_tensors = dict(self.model.loss_terms['test'], **self.model.metrics['test'])
        self._tensors_to_evaluate = dict([(n, t) for n, t in all_tensors.items()])
        loss_terms_to_evaluate = dict([(n, t) for n, t in self.model.loss_terms['test'].items()
                                       if t in self._tensors_to_evaluate.values()])
        metrics_to_evaluate = dict([(n, t) for n, t in self.model.metrics['test'].items()
                                    if t in self._tensors_to_evaluate.values()])

        # Placeholders for writing summaries at end of test run
        self._placeholders = {}
        for type_, tensors in (('loss', loss_terms_to_evaluate),
                               ('metric', metrics_to_evaluate)):
            for name in tensors.keys():
                name = '%s/test/%s' % (type_, name)
                placeholder = tf.placeholder(dtype=np.float32, name=name + '_placeholder')
                self.summary.scalar(name, placeholder)
                self._placeholders[name.split('/')[-1]] = placeholder

    def copy_model_weights(self):
        """Copy weights from main model used for training.

        This operation should stop-the-world, that is, training should not occur.
        """
        assert self._copy_variables_to_test_model_op is not None
        self._tensorflow_session.run(self._copy_variables_to_test_model_op)
        logger.debug('Copied over trainable model parameters for testing.')

"""Default specification of a data source."""
class BaseDataSource(object):
    """Base DataSource class."""

    def __init__(self,
                 tensorflow_session: tf.compat.v1.Session,
                 data_format: str = 'NHWC',
                 batch_size: int = 32,
                 num_threads: int = max(4, multiprocessing.cpu_count()),
                 min_after_dequeue: int = 1000,
                 fread_queue_capacity: int = 0,
                 preprocess_queue_capacity: int = 0,
                 staging=False,
                 shuffle=None,
                 testing=False,
                 ):
        """Initialize a data source instance."""
        assert tensorflow_session is not None and isinstance(tensorflow_session, tf.Session)
        assert isinstance(batch_size, int) and batch_size > 0
        if shuffle is None:
            shuffle = staging
        self.testing = testing
        if testing:
            assert not shuffle and not staging
            # if num_threads != 1:
            #     logger.info('Forcing use of single thread for live testing.')
            # num_threads = 1
        self.staging = staging
        self.shuffle = shuffle
        self.data_format = data_format.upper()
        assert self.data_format == 'NHWC' or self.data_format == 'NCHW'
        self.batch_size = batch_size
        self.num_threads = num_threads
        self._tensorflow_session = tensorflow_session
        self._coordinator = tf.train.Coordinator()
        self.all_threads = []

        # Setup file read queue
        self._fread_queue_capacity = fread_queue_capacity
        if self._fread_queue_capacity == 0:
            self._fread_queue_capacity = (num_threads + 1) * batch_size
        self._fread_queue = queue.Queue(maxsize=self._fread_queue_capacity)

        with tf.variable_scope(''.join(c for c in self.short_name if c.isalnum())):
            # Setup preprocess queue
            labels, dtypes, shapes = self._determine_dtypes_and_shapes()
            self._preprocess_queue_capacity = (min_after_dequeue + (num_threads + 1) * batch_size
                                               if preprocess_queue_capacity == 0
                                               else preprocess_queue_capacity)
            if shuffle:
                self._preprocess_queue = tf.RandomShuffleQueue(
                        capacity=self._preprocess_queue_capacity,
                        min_after_dequeue=min_after_dequeue,
                        dtypes=dtypes, shapes=shapes,
                )
            else:
                self._preprocess_queue = tf.FIFOQueue(
                        capacity=self._preprocess_queue_capacity,
                        dtypes=dtypes, shapes=shapes,
                )
            self._tensors_to_enqueue = OrderedDict([
                (label, tf.placeholder(dtype, shape=shape, name=label))
                for label, dtype, shape in zip(labels, dtypes, shapes)
            ])

            self._enqueue_op = \
                self._preprocess_queue.enqueue(tuple(self._tensors_to_enqueue.values()))
            self._preprocess_queue_close_op = \
                self._preprocess_queue.close(cancel_pending_enqueues=True)
            self._preprocess_queue_size_op = self._preprocess_queue.size()
            self._preprocess_queue_clear_op = \
                self._preprocess_queue.dequeue_up_to(self._preprocess_queue.size())
                
            if not staging:
                output_tensors = self._preprocess_queue.dequeue_many(self.batch_size)
                if not isinstance(output_tensors, list):
                    output_tensors = [output_tensors]
                self._output_tensors = dict([
                    (label, tensor) for label, tensor in zip(labels, output_tensors)
                ])
            else:
                # Setup on-GPU staging area
                self._staging_area = tf.contrib.staging.StagingArea(
                    dtypes=dtypes,
                    shapes=[tuple([batch_size] + list(shape)) for shape in shapes],
                    capacity=1,  # This does not have to be high
                )
                self._staging_area_put_op = \
                    self._staging_area.put(self._preprocess_queue.dequeue_many(batch_size))
                self._staging_area_clear_op = self._staging_area.clear()

                self._output_tensors = dict([
                    (label, tensor) for label, tensor in zip(labels, self._staging_area.get())
                ])
            
        logger.info('Initialized data source: "%s"' % self.short_name)

    def __del__(self):
        """Destruct and clean up instance."""
        self.cleanup()

    @property
    def num_entries(self):
        """Number of entries in this data source.

        Used to calculate number of steps to train when asked to be trained for # epochs.
        """
        raise NotImplementedError('BaseDataSource::num_entries not specified.')

    @property
    def short_name(self):
        """Short identifier for data source.

        Overload this magic method if the class is generic, eg. supporting h5py/numpy arrays as
        input with specific data sources.
        """
        raise NotImplementedError('BaseDataSource::short_name not specified.')

    __cleaned_up = False

    def cleanup(self):
        """Force-close all threads."""
        if self.__cleaned_up:
            return

        # Clear queues
        fread_threads = [t for t in self.all_threads if t.name.startswith('fread_')]
        preprocess_threads = [t for t in self.all_threads if t.name.startswith('preprocess_')]
        transfer_threads = [t for t in self.all_threads if t.name.startswith('transfer_')]

        self._coordinator.request_stop()

        # Unblock any self._fread_queue.put calls
        while True:
            try:
                self._fread_queue.get_nowait()
            except queue.Empty:
                break
            time.sleep(0.1)

        # Push data through to trigger exits in preprocess/transfer threads
        for _ in range(self.batch_size * self.num_threads):
            self._fread_queue.put(None)
        self._tensorflow_session.run(self._preprocess_queue_close_op)
        if self.staging:
            self._tensorflow_session.run(self._staging_area_clear_op)

        self._coordinator.join(self.all_threads, stop_grace_period_secs=5)
        self.__cleaned_up = True

    def reset(self):
        """Reset threads and empty queues (where possible)."""
        assert self.testing is True

        # Clear queues
        self._coordinator.request_stop()
        with self._fread_queue.mutex:  # Unblock any self._fread_queue.get calls
            self._fread_queue.queue.clear()
        for _ in range(2*self.num_threads):
            self._fread_queue.put(None)
        while True:  # Unblock any enqueue requests
            preprocess_queue_size = self._tensorflow_session.run(self._preprocess_queue_size_op)
            if preprocess_queue_size == 0:
                break
            self._tensorflow_session.run(self._preprocess_queue_clear_op)
            time.sleep(0.1)
        while True:  # Unblock any self._fread_queue.put calls
            try:
                self._fread_queue.get_nowait()
            except queue.Empty:
                break
            time.sleep(0.1)
        self._coordinator.join(self.all_threads, stop_grace_period_secs=5)

        # Restart threads
        self._coordinator.clear_stop()
        self.create_and_start_threads()

    def _determine_dtypes_and_shapes(self):
        """Determine the dtypes and shapes of Tensorflow queue and staging area entries."""
        while True:
            raw_entry = next(self.entry_generator(yield_just_one=True))
            if raw_entry is None:
                continue
            preprocessed_entry_dict = self.preprocess_entry(raw_entry)
            if preprocessed_entry_dict is not None:
                break
        labels, values = zip(*list(preprocessed_entry_dict.items()))
        dtypes = [value.dtype for value in values]
        shapes = [value.shape for value in values]
        return labels, dtypes, shapes

    def entry_generator(self, yield_just_one=False):
        """Return a generator which reads an entry from disk or memory.

        This method should be thread-safe so make sure to use threading.Lock where necessary.
        The implemented method should explicitly handle the `yield_just_one=True` case to only
        yield one entry without hanging in the middle of an infinite loop.
        """
        raise NotImplementedError('BaseDataSource::entry_generator not implemented.')

    def preprocess_entry(self, entry):
        """Preprocess a "raw" data entry and yield a dict.

        Each element of an entry is provided to this method as separate arguments.
        This method should be thread-safe so make sure to use threading.Lock where necessary.
        """
        raise NotImplementedError('BaseDataSource::preprocess_entry not implemented.')

    def read_entry_job(self):
        """Job to read an entry and enqueue to _fread_queue."""
        read_entry = self.entry_generator()
        while not self._coordinator.should_stop():
            try:
                entry = next(read_entry)
            except StopIteration:
                if not self.testing:
                    continue
                else:
                    logger.debug('Reached EOF in %s' % threading.current_thread().name)
                    break
            if entry is not None:
                self._fread_queue.put(entry)
        read_entry.close()
        logger.debug('Exiting thread %s' % threading.current_thread().name)

    def preprocess_job(self):
        """Job to fetch and preprocess an entry."""
        while not self._coordinator.should_stop():
            raw_entry = self._fread_queue.get()
            if raw_entry is None:
                return
            preprocessed_entry_dict = self.preprocess_entry(raw_entry)
            if preprocessed_entry_dict is not None:
                feed_dict = dict([(self._tensors_to_enqueue[label], value)
                                  for label, value in preprocessed_entry_dict.items()])
                try:
                    self._tensorflow_session.run(self._enqueue_op, feed_dict=feed_dict)
                except (tf.errors.CancelledError, RuntimeError):
                    break
        logger.debug('Exiting thread %s' % threading.current_thread().name)

    def transfer_to_gpu_job(self):
        """Transfer a data entry from CPU memory to GPU memory."""
        while not self._coordinator.should_stop():
            try:
                self._tensorflow_session.run(self._staging_area_put_op)
            except tf.errors.CancelledError or tf.errors.OutOfRangeError:
                break
        logger.debug('Exiting thread %s' % threading.current_thread().name)

    def create_threads(self):
        """Create Python threads for multi-threaded read and preprocess jobs."""
        name = self.short_name
        self.all_threads = []
        tf.train.add_queue_runner(tf.train.QueueRunner(self._preprocess_queue, [self._enqueue_op] * 2))

        def _create_and_register_thread(*args, **kwargs):
            thread = threading.Thread(*args, **kwargs)
            thread.daemon = True
            self.all_threads.append(thread)

        for i in range(self.num_threads):
            # File read thread
            _create_and_register_thread(target=self.read_entry_job, name='fread_%s_%d' % (name, i))

            # Preprocess thread
            _create_and_register_thread(target=self.preprocess_job,
                                        name='preprocess_%s_%d' % (name, i))

        if self.staging:
            # Send-to-GPU thread
            _create_and_register_thread(target=self.transfer_to_gpu_job,
                                        name='transfer_%s_%d' % (name, i))

    def start_threads(self):
        """Begin executing all created threads."""
        assert len(self.all_threads) > 0
        for thread in self.all_threads:
            thread.start()

    def create_and_start_threads(self):
        """Create and begin threads for preprocessing."""
        self.create_threads()
        self.start_threads()

    @property
    def output_tensors(self):
        """Return tensors holding a preprocessed batch."""
        return self._output_tensors


"""Base model class for Tensorflow-based model construction."""
class BaseModel(object):
    """Base model class for Tensorflow-based model construction.

    This class assumes that there exist no other Tensorflow models defined.
    That is, any variable that exists in the Python session will be grabbed by the class.
    """

    def __init__(self,
                 tensorflow_session: tf.compat.v1.Session,
                 learning_schedule: List[Dict[str, Any]] = [],
                 train_data: Dict[str, BaseDataSource] = {},
                 test_data: Dict[str, BaseDataSource] = {},
                 test_losses_or_metrics: str = None,
                 use_batch_statistics_at_test: bool = True,
                 identifier: str = None):
        """Initialize model with data sources and parameters."""
        self._tensorflow_session = tensorflow_session
        self._train_data = train_data
        self._test_data = test_data
        self._test_losses_or_metrics = test_losses_or_metrics
        self._initialized = False
        self.__identifier = identifier

        # Extract and keep known prefixes/scopes
        self._learning_schedule = learning_schedule
        self._known_prefixes = [schedule for schedule in learning_schedule]

        # Check consistency of given data sources
        train_data_sources = list(train_data.values())
        test_data_sources = list(test_data.values())
        all_data_sources = train_data_sources + test_data_sources
        first_data_source = all_data_sources.pop()
        self._batch_size = first_data_source.batch_size
        self._data_format = first_data_source.data_format
        for data_source in all_data_sources:
            if data_source.batch_size != self._batch_size:
                raise ValueError(('Data source "%s" has anomalous batch size of %d ' +
                                  'when detected batch size is %d.') % (data_source.short_name,
                                                                        data_source.batch_size,
                                                                        self._batch_size))
            if data_source.data_format != self._data_format:
                raise ValueError(('Data source "%s" has anomalous data_format of %s ' +
                                  'when detected data_format is %s.') % (data_source.short_name,
                                                                         data_source.data_format,
                                                                         self._data_format))
        self._data_format_longer = ('channels_first' if self._data_format == 'NCHW'
                                    else 'channels_last')

        # Make output dir
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

        # Log messages to file
        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(self.output_path + '/messages.log')
        file_handler.setFormatter(root_logger.handlers[0].formatter)
        for handler in root_logger.handlers[1:]:  # all except stdout
            root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)

        # Register a manager for tf.Summary
        self.summary = SummaryManager(self)

        # Register a manager for checkpoints
        self.checkpoint = CheckpointManager(self)

        # Register a manager for timing related operations
        self.time = TimeManager(self)

        # Prepare for live (concurrent) validation/testing during training, on the CPU
        self._enable_live_testing = (len(self._train_data) > 0) and (len(self._test_data) > 0)
        self._tester = LiveTester(self, self._test_data, use_batch_statistics_at_test)

        # Run-time parameters
        with tf.variable_scope('learning_params'):
            self.is_training = tf.placeholder(tf.bool)
            self.use_batch_statistics = tf.placeholder(tf.bool)
            self.learning_rate_multiplier = tf.Variable(1.0, trainable=False, dtype=tf.float32)
            self.learning_rate_multiplier_placeholder = tf.placeholder(dtype=tf.float32)
            self.assign_learning_rate_multiplier = \
                tf.assign(self.learning_rate_multiplier, self.learning_rate_multiplier_placeholder)

        self._build_all_models()

    def __del__(self):
        """Explicitly call methods to cleanup any live threads."""
        train_data_sources = list(self._train_data.values())
        test_data_sources = list(self._test_data.values())
        all_data_sources = train_data_sources + test_data_sources
        for data_source in all_data_sources:
            data_source.cleanup()
        self._tester.__del__()

    __identifier_stem = None

    @property
    def identifier(self):
        """Identifier for model based on time."""
        if self.__identifier is not None:  # If loading from checkpoints or having naming enforced
            return self.__identifier
        if self.__identifier_stem is None:
            self.__identifier_stem = self.__class__.__name__ + '/' + time.strftime('%y%m%d%H%M%S')
        return self.__identifier_stem + self._identifier_suffix

    @property
    def _identifier_suffix(self):
        """Identifier suffix for model based on data sources and parameters."""
        return ''

    @property
    def output_path(self):
        """Path to store logs and model weights into."""
        return '%s/%s' % (os.path.abspath(os.path.dirname(__file__) + 'outputs'),
                          self.identifier)

    def _build_all_models(self):
        """Build training (GPU/CPU) and testing (CPU) streams."""
        self.output_tensors = {}
        self.loss_terms = {}
        self.metrics = {}

        def _build_datasource_summaries(data_sources, mode):
            """Register summary operations for input data from given data sources."""
            with tf.variable_scope('%s_data' % mode):
                for data_source_name, data_source in data_sources.items():
                    tensors = data_source.output_tensors
                    for key, tensor in tensors.items():
                        summary_name = '%s/%s' % (data_source_name, key)
                        shape = tensor.shape.as_list()
                        num_dims = len(shape)
                        if num_dims == 4:  # Image data
                            if shape[1] == 1 or shape[1] == 3:
                                self.summary.image(summary_name, tensor,
                                                   data_format='channels_first')
                            elif shape[3] == 1 or shape[3] == 3:
                                self.summary.image(summary_name, tensor,
                                                   data_format='channels_last')
                            # TODO: fix issue with no summary otherwise
                        elif num_dims == 2:
                            self.summary.histogram(summary_name, tensor)
                        else:
                            logger.debug('I do not know how to create a summary for %s (%s)' %
                                         (summary_name, tensor.shape.as_list()))

        def _build_train_or_test(mode):
            data_sources = self._train_data if mode == 'train' else self._test_data

            # Build model
            output_tensors, loss_terms, metrics = self.build_model(data_sources, mode=mode)

            # Record important tensors
            self.output_tensors[mode] = output_tensors
            self.loss_terms[mode] = loss_terms
            self.metrics[mode] = metrics

            # Create summaries for scalars
            if mode == 'train':
                for name, loss_term in loss_terms.items():
                    self.summary.scalar('loss/%s/%s' % (mode, name), loss_term)
                for name, metric in metrics.items():
                    self.summary.scalar('metric/%s/%s' % (mode, name), metric)

        # Build the main model
        if len(self._train_data) > 0:
            _build_datasource_summaries(self._train_data, mode='train')
            _build_train_or_test(mode='train')
            logger.info('Built model.')

            # Print no. of parameters and lops
            flops = tf.profiler.profile(
                options=tf.profiler.ProfileOptionBuilder(
                    tf.profiler.ProfileOptionBuilder.float_operation()
                ).with_empty_output().build())
            logger.info('------------------------------')
            logger.info(' Approximate Model Statistics ')
            logger.info('------------------------------')
            logger.info('FLOPS per input: {:,}'.format(flops.total_float_ops / self._batch_size))
            logger.info(
                'Trainable Parameters: {:,}'.format(
                    np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
                )
            )
            logger.info('------------------------------')

        # If there are any test data streams, build same model with different scope
        # Trainable parameters will be copied at test time
        if len(self._test_data) > 0:
            _build_datasource_summaries(self._test_data, mode='test')
            with tf.variable_scope('test'):
                _build_train_or_test(mode='test')
            logger.info('Built model for live testing.')

        if self._enable_live_testing:
            self._tester._post_model_build()  # Create copy ops to be run before every test run

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        raise NotImplementedError('BaseModel::build_model is not yet implemented.')

    def initialize_if_not(self, training=False):
        """Initialize variables and begin preprocessing threads."""
        if self._initialized:
            return

        # Build supporting operations
        with tf.variable_scope('savers'):
            self.checkpoint.build_savers()  # Create savers
        if training:
            with tf.variable_scope('optimize'):
                self._build_optimizers()

        # Start pre-processing routines
        for _, datasource in self._train_data.items():
            datasource.create_and_start_threads()

        # Initialize all variables
        self._tensorflow_session.run(tf.global_variables_initializer())
        self._initialized = True

    def _build_optimizers(self):
        """Based on learning schedule, create optimizer instances."""
        self._optimize_ops = []
        all_trainable_variables = tf.trainable_variables()
        all_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        all_reg_losses = tf.losses.get_regularization_losses()
        for spec in self._learning_schedule:
            optimize_ops = []
            update_ops = []
            loss_terms = spec['loss_terms_to_optimize']
            reg_losses = []
            assert isinstance(loss_terms, dict)
            for loss_term_key, prefixes in loss_terms.items():
                assert loss_term_key in self.loss_terms['train'].keys()
                variables_to_train = []
                for prefix in prefixes:
                    variables_to_train += [
                        v for v in all_trainable_variables
                        if v.name.startswith(prefix)
                    ]
                    update_ops += [
                        o for o in all_update_ops
                        if o.name.startswith(prefix)
                    ]
                    reg_losses += [
                        l for l in all_reg_losses
                        if l.name.startswith(prefix)
                    ]

                optimizer_class = tf.train.AdamOptimizer
                optimizer = optimizer_class(
                    learning_rate=self.learning_rate_multiplier * spec['learning_rate'],
                    # beta1=0.9,
                    # beta2=0.999,
                )
                final_loss = self.loss_terms['train'][loss_term_key]
                if len(reg_losses) > 0:
                    final_loss += tf.reduce_sum(reg_losses)
                with tf.control_dependencies(update_ops):
                    gradients, variables = zip(*optimizer.compute_gradients(
                        loss=final_loss,
                        var_list=variables_to_train,
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
                    ))
                    # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)  # TODO: generalize
                    optimize_op = optimizer.apply_gradients(zip(gradients, variables))
                optimize_ops.append(optimize_op)
            self._optimize_ops.append(optimize_ops)
            logger.info('Built optimizer for: %s' % ', '.join(loss_terms.keys()))

    def train_loop_pre(self, current_step):
        """Run this at beginning of training loop."""
        pass

    def train_loop_post(self, current_step):
        """Run this at end of training loop."""
        pass

    def train(self, num_epochs=None, num_steps=None):
        """Train model as requested."""
        if num_steps is None:
            num_entries = np.min([s.num_entries for s in list(self._train_data.values())])
            num_steps = int(num_epochs * num_entries / self._batch_size)
        self.initialize_if_not(training=True)
        
        # ==================================== alonsag monitoring ====================================
        heatmap_monitor = []
        radius_monitor = []
        # ============================================================================================

        try:
            initial_step = self.checkpoint.load_all()
            current_step = initial_step
            
            print("-I-", num_steps-initial_step, "steps are planned")
            for current_step in range(initial_step, num_steps):
                # Extra operations defined in implementation of this base class
                self.train_loop_pre(current_step)

                # Select loss terms, optimize operations, and metrics tensors to evaluate
                fetches = {}
                schedule_id = current_step % len(self._learning_schedule)
                schedule = self._learning_schedule[schedule_id]
                fetches['optimize_ops'] = self._optimize_ops[schedule_id]
                loss_term_keys, _ = zip(*list(schedule['loss_terms_to_optimize'].items()))
                fetches['loss_terms'] = [self.loss_terms['train'][k] for k in loss_term_keys]
                summary_op = self.summary.get_ops(mode='train')
                if len(summary_op) > 0:
                    fetches['summaries'] = summary_op

                # Run one optimization iteration and retrieve calculated loss values
                self.time.start('train_iteration', average_over_last_n_timings=100)
                outcome = self._tensorflow_session.run(
                    fetches=fetches,
                    feed_dict={
                        self.is_training: True,
                        self.use_batch_statistics: True,
                    }
                )
                self.time.end('train_iteration')

                # Print progress
                to_print = '%07d> ' % current_step
                to_print += ', '.join(['%s = %g' % (k, v)
                                       for k, v in zip(loss_term_keys, outcome['loss_terms'])])
                self.time.log_every('train_iteration', to_print, seconds=2)
                
                # ==================================== alonsag monitoring ====================================
                heatmap_monitor.append(outcome['loss_terms'][0])
                radius_monitor.append(outcome['loss_terms'][1])
                # ============================================================================================
                # Trigger copy weights & concurrent testing (if not already running)
                if self._enable_live_testing:
                    self._tester.trigger_test_if_not_testing(current_step)

                # Write summaries
                if 'summaries' in outcome:
                    self.summary.write_summaries(outcome['summaries'], current_step)

                # Save model weights
                if self.time.has_been_n_seconds_since_last('save_weights', 300) \
                        and current_step > initial_step:
                    self.checkpoint.save_all(current_step)

                # Extra operations defined in implementation of this base class
                self.train_loop_post(current_step)  
        except KeyboardInterrupt:
            # Handle CTRL-C graciously
            self.checkpoint.save_all(current_step)
            sys.exit(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(heatmap_monitor)
        ax1.set_title('Heatmap MSE Over Time')
        ax2.plot(radius_monitor)
        ax2.set_title('Radius MSE Over Time')
        plt.show()
        # Stop live testing, and run final full test
        if self._enable_live_testing:
            self._tester.do_final_full_test(current_step)

        # Save final weights
        if current_step > initial_step:
            self.checkpoint.save_all(current_step)

    def inference_generator(self):
        """Perform inference on test data and yield a batch of output."""
        self.initialize_if_not(training=False)
        self.checkpoint.load_all()  # Load available weights

        # TODO: Make more generic by not picking first source
        data_source = next(iter(self._train_data.values()))
        while True:
            fetches = dict(self.output_tensors['train'], **data_source.output_tensors)
            start_time = time.time()
            outputs = self._tensorflow_session.run(
                fetches=fetches,
                feed_dict={
                    self.is_training: False,
                    self.use_batch_statistics: True,
                },
            )
            outputs['inference_time'] = 1e3*(time.time() - start_time)
            yield outputs

"""Manage registration and evaluation of summary operations."""
class SummaryManager(object):
    """Manager to remember and run summary operations as necessary."""

    def __init__(self, model, cheap_ops_every_n_secs=2, expensive_ops_every_n_mins=2):
        """Initialize manager based on given model instance."""
        self._tensorflow_session = model._tensorflow_session
        self._model = model
        self._cheap_ops = {
            'train': {},
            'test': {},
            'full_test': {},
        }
        self._expensive_ops = {
            'train': {},
            'test': {},
            'full_test': {},
        }
        self._cheap_ops_every_n_secs = cheap_ops_every_n_secs
        self._expensive_ops_every_n_secs = 60 * expensive_ops_every_n_mins

        self._ready_to_write = False

    def _prepare_for_write(self):
        """Merge together cheap and expensive ops separately."""
        self._writer = tf.summary.FileWriter(self._model.output_path,
                                             self._tensorflow_session.graph)
        for mode in ('train', 'test', 'full_test'):
            self._expensive_ops[mode].update(self._cheap_ops[mode])
        self._ready_to_write = True

    def get_ops(self, mode='train'):
        """Retrieve summary ops to evaluate at given iteration number."""
        if not self._ready_to_write:
            self._prepare_for_write()
        if mode == 'test' or mode == 'full_test':  # Always return all ops for test case
            return self._expensive_ops[mode]
        elif mode == 'train':  # Select ops to evaluate based on defined frequency
            check_func = self._model.time.has_been_n_seconds_since_last
            if check_func('expensive_summaries_train', self._expensive_ops_every_n_secs):
                return self._expensive_ops[mode]
            elif check_func('cheap_summaries_train', self._cheap_ops_every_n_secs):
                return self._cheap_ops[mode]
        return {}

    def write_summaries(self, summary_outputs, iteration_number):
        """Write given outputs to `self._writer`."""
        for _, summary in summary_outputs.items():
            self._writer.add_summary(summary, global_step=iteration_number)

    def _get_clean_name(self, operation):
        name = operation.name

        # Determine mode
        mode = 'train'
        if name.startswith('test/') or name.startswith('test_data/'):
            mode = 'test'
        elif name.startswith('loss/test/') or name.startswith('metric/test/'):
            mode = 'full_test'

        # Correct name
        if mode == 'test':
            name = name[name.index('/') + 1:]
        elif mode == 'full_test':
            name = '/'.join(name.split('/')[2:])
        if name[-2] == ':':
            name = name[:-2]
        return mode, name

    def _register_cheap_op(self, operation):
        mode, name = self._get_clean_name(operation)
        try:
            assert name not in self._cheap_ops[mode] and name not in self._expensive_ops[mode]
        except AssertionError:
            raise Exception('Duplicate definition of summary item: "%s"' % name)
        self._cheap_ops[mode][name] = operation

    def _register_expensive_op(self, operation):
        mode, name = self._get_clean_name(operation)
        try:
            assert name not in self._cheap_ops[mode] and name not in self._expensive_ops[mode]
        except AssertionError:
            raise Exception('Duplicate definition of summary item: "%s"' % name)
        self._expensive_ops[mode][name] = operation

    def audio(self, name, tensor, **kwargs):
        """TODO: Log summary of audio."""
        raise NotImplementedError('SummaryManager::audio not implemented.')

    def text(self, name, tensor, **kwargs):
        """TODO: Log summary of text."""
        raise NotImplementedError('SummaryManager::text not implemented.')

    def histogram(self, name, tensor, **kwargs):
        """TODO: Log summary of audio."""
        operation = tf.summary.histogram(name, tensor, **kwargs)
        self._register_expensive_op(operation)

    def image(self, name, tensor, data_format='channels_last', **kwargs):
        """TODO: Log summary of image."""
        if data_format == 'channels_first':
            tensor = tf.transpose(tensor, perm=(0, 2, 3, 1))
        c = tensor.shape.as_list()[-1]
        if c == 3:  # Assume RGB and convert to BGR for visualization
            tensor = tensor[:, :, :, ::-1]   # TODO: find better solution
        operation = tf.summary.image(name, tensor, **kwargs)
        self._register_expensive_op(operation)

    def _4d_tensor(self, name, tensor, **kwargs):
        """Display all filters in a grid for visualization."""
        h, w, c, num_tensor = tensor.shape.as_list()

        # Try to visualise convolutional filters or feature maps
        # See: https://gist.github.com/kukuruza/03731dc494603ceab0c5
        # input shape: (Y, X, C, N)
        if c != 1 and c != 3:
            tensor = tf.reduce_mean(tensor, axis=2, keep_dims=True)
            c = 1
        # shape is now: (Y, X, 1|C, N)
        v_min = tf.reduce_min(tensor)
        v_max = tf.reduce_max(tensor)
        tensor -= v_min
        tensor *= 1.0 / (v_max - v_min)
        tensor = tf.pad(tensor, [[1, 0], [1, 0], [0, 0], [0, 0]], 'CONSTANT')
        tensor = tf.transpose(tensor, perm=(3, 0, 1, 2))
        # shape is now: (N, Y, X, C)
        # place tensor on grid
        num_tensor_x = int(np.round(np.sqrt(num_tensor)))
        num_tensor_y = num_tensor / num_tensor_x
        while not num_tensor_y.is_integer():
            num_tensor_x += 1
            num_tensor_y = num_tensor / num_tensor_x
        num_tensor_y = int(num_tensor_y)
        h += 1
        w += 1
        tensor = tf.reshape(tensor, (num_tensor_x, h * num_tensor_y, w, c))
        # shape is now: (N_x, Y * N_y, X, c)
        tensor = tf.transpose(tensor, (0, 2, 1, 3))
        # shape is now: (N_x, X, Y * N_y, c)
        tensor = tf.reshape(tensor, (1, w * num_tensor_x, h * num_tensor_y, c))
        # shape is now: (1, X * N_x, Y * N_y, c)
        tensor = tf.transpose(tensor, (0, 2, 1, 3))
        # shape is now: (1, Y * N_y, X * N_x, c)
        tensor = tf.pad(tensor, [[0, 0], [0, 1], [0, 1], [0, 0]], 'CONSTANT')

        self.image(name, tensor, **kwargs)

    def filters(self, name, tensor, **kwargs):
        """Log summary of convolutional filters.

        Note that this method expects the output of the convolutional layer when using
        `tf.layers.conv2d` or for the filters to be defined in the same scope as the output tensor.
        """
        assert 'data_format' not in kwargs
        with tf.name_scope('viz_filters'):
            # Find tensor holding trainable kernel weights
            name_stem = '/'.join(tensor.name.split('/')[:-1]) + '/kernel'
            matching_tensors = [t for t in tf.trainable_variables() if t.name.startswith(name_stem)]
            assert len(matching_tensors) == 1
            filters = matching_tensors[0]

            # H x W x C x N
            h, w, c, n = filters.shape.as_list()
            filters = tf.transpose(filters, perm=(3, 2, 0, 1))
            # N x C x H x W
            filters = tf.reshape(filters, (n*c, 1, h, w))
            # NC x 1 x H x W
            filters = tf.transpose(filters, perm=(2, 3, 1, 0))
            # H x W x 1 x NC

            self._4d_tensor(name, filters, **kwargs)

    def feature_maps(self, name, tensor, mean_across_channels=True, data_format='channels_last',
                     **kwargs):
        """Log summary of feature maps / image activations."""
        with tf.name_scope('viz_featuremaps'):
            if data_format == 'channels_first':
                # N x C x H x W
                tensor = tf.transpose(tensor, perm=(0, 2, 3, 1))
            # N x H x W x C
            if mean_across_channels:
                tensor = tf.reduce_mean(tensor, axis=3, keepdims=True)
                # N x H x W x 1
                tensor = tf.transpose(tensor, perm=(1, 2, 3, 0))
            else:
                n, c, h, w = tensor.shape.as_list()
                tensor = tf.reshape(tensor, (n*c, 1, h, w))
                # N x 1 x H x W
                tensor = tf.transpose(tensor, perm=(2, 3, 1, 0))
            # H x W x 1 x N

            self._4d_tensor(name, tensor, **kwargs)

    def tiled_images(self, name, tensor, data_format='channels_last', **kwargs):
        """Log summary of feature maps / image activations."""
        with tf.name_scope('viz_featuremaps'):
            if data_format == 'channels_first':
                # N x C x H x W
                tensor = tf.transpose(tensor, perm=(0, 2, 3, 1))
            # N x H x W x C
            tensor = tf.transpose(tensor, perm=(1, 2, 3, 0))
            # H x W x C x N
            self._4d_tensor(name, tensor, **kwargs)

    def scalar(self, name, tensor, **kwargs):
        """Log summary of scalar."""
        operation = tf.summary.scalar(name, tensor, **kwargs)
        self._register_cheap_op(operation)

"""Manage saving and loading of model checkpoints."""
class CheckpointManager(object):
    """Manager to coordinate saving and loading of trainable parameters."""

    def __init__(self, model):
        """Initialize manager based on given model instance."""
        self._tensorflow_session = model._tensorflow_session
        self._model = model

    def build_savers(self):
        """Create tf.train.Saver instances."""
        all_saveable_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) +
                                   tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS) +
                                   tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES) +
                                   tf.get_collection_ref('batch_norm_non_trainable'),
                                   key=lambda v: v.name)

        # Grab all available prefixes
        all_prefixes = []
        for v in all_saveable_vars:
            name = v.name
            if '/' not in name:
                continue
            prefix = name.split('/')[0]
            if prefix == 'test' or prefix == 'learning_params':
                continue
            if prefix not in all_prefixes:
                all_prefixes.append(prefix)

        # For each prefix, create saver
        self._savers = {}
        for prefix in all_prefixes:
            vars_to_save = [v for v in all_saveable_vars if v.name.startswith(prefix + '/')]
            if len(vars_to_save):
                self._savers[prefix] = tf.train.Saver(vars_to_save, max_to_keep=2)

    def load_all(self):
        """Load all available weights for each known prefix."""
        iteration_number = 0
        iteration_numbers = []
        for prefix, saver in self._savers.items():
            output_path = '%s/checkpoints/%s' % (self._model.output_path, prefix)
            checkpoint = tf.train.get_checkpoint_state(output_path)
            if checkpoint and checkpoint.model_checkpoint_path:
                checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
                try:  # Attempt to restore saveable variables
                    self._savers[prefix].restore(self._tensorflow_session,
                                                 '%s/%s' % (output_path, checkpoint_name))
                    iteration_numbers.append(
                        int(next(re.finditer("(\d+)(?!.*\d)", checkpoint_name)).group(0))
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
        if len(iteration_numbers) > 0:
            iteration_number = np.amax(iteration_numbers)
        return iteration_number

    def save_all(self, iteration_number):
        """Save all prefixes."""
        prefixes_to_use = []
        for schedule in self._model._learning_schedule:
            for prefixes in schedule['loss_terms_to_optimize'].values():
                prefixes_to_use += prefixes
        prefixes_to_use = list(set(prefixes_to_use))

        for prefix, saver in self._savers.items():
            if prefix not in prefixes_to_use:
                continue
            output_path = '%s/checkpoints/%s' % (self._model.output_path, prefix)
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            saver.save(self._tensorflow_session, output_path + '/model',
                       global_step=iteration_number)
            logger.debug('Saved %s' % output_path)
        logger.info('CheckpointManager::save_all call done')

"""Routines to time events and restrict logs or operations by frequency."""
class TimeManager(object):
    """Manage timing of event executions or measure timings."""

    def __init__(self, model):
        """Initialize manager based on given model instance."""
        self._tensorflow_session = model._tensorflow_session
        self._model = model

        self._timers = {}
        self._last_time = {}

    def start(self, name, **kwargs):
        """Begin timer for given event/operation."""
        if name not in self._timers:
            timer = Timer(**kwargs)
            self._timers[name] = timer
        else:
            timer = self._timers[name]
        timer.start()

    def end(self, name):
        """End timer for given event/operation."""
        assert name in self._timers
        return self._timers[name].end()

    def has_been_n_seconds_since_last(self, identifier, seconds):
        """Indicate if enough time has passed since last time.

        Also updates the `last time` record based on identifier.
        """
        current_time = time.time()
        if identifier not in self._last_time or \
           (current_time - self._last_time[identifier] > seconds):
            self._last_time[identifier] = current_time
            return True
        return False

    def log_every(self, identifier, message, seconds=1):
        """Limit logging of messages based on specified interval and identifier."""
        if self.has_been_n_seconds_since_last(identifier, seconds):
            logger.info(message)
        else:
            logger.debug(message)


# A local Timer class for timing
class Timer(object):
    """Record start and end times as requested and provide summaries."""

    def __init__(self, average_over_last_n_timings=10):
        """Store keyword parameters."""
        self._average_over_last_n_timings = average_over_last_n_timings
        self._active = False
        self._timings = []
        self._start_time = -1

    def start(self):
        """Cache starting time."""
        # assert not self._active
        self._start_time = time.time()
        self._active = True

    def end(self):
        """Check ending time and store difference."""
        assert self._active and self._start_time > 0

        # Calculate difference
        end_time = time.time()
        time_difference = end_time - self._start_time

        # Record timing (and trim history)
        self._timings.append(time_difference)
        if len(self._timings) > self._average_over_last_n_timings:
            self._timings = self._timings[-self._average_over_last_n_timings:]

        # Reset
        self._start_time = -1
        self._active = False

        return time_difference

    @property
    def current_mean(self):
        """Calculate mean timing for as many trials as specified in constructor."""
        values = self._timings
        return np.mean(values)
